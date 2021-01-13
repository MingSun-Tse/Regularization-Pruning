import torch
import torch.nn as nn
import torch.optim as optim
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
from utils import plot_weights_heatmap
import matplotlib.pyplot as plt
pjoin = os.path.join

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self.hist_mag_ratio = {}
        self.n_update_reg = {}
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.original_w_mag = {}
        self.original_kept_w_mag = {}
        self.ranking = {}
        self.pruned_wg_L1 = {}
        self.all_layer_finish_pick = False
        self.w_abs = {}
        self.mag_reg_log = {}
        if self.args.__dict__.get('AdaReg_only_picking'): # AdaReg is the old name for GReg-2
            self.original_model = copy.deepcopy(self.model)
        
        # prune_init, to determine the pruned weights
        # this will update the 'self.kept_wg' and 'self.pruned_wg' 
        if self.args.method in ['GReg-1', 'GReg-2']:
            self._get_kept_wg_L1()
        for k, v in self.pruned_wg.items():
            self.pruned_wg_L1[k] = v
        if self.args.method == 'GReg-2': # GReg-2 will determine which wgs to prune later, so clear it here
            self.kept_wg = {}
            self.pruned_wg = {}

        self.prune_state = "update_reg"
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                shape = m.weight.data.shape

                # initialize reg
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    self.reg[name] = torch.zeros(shape[0], shape[1]).cuda() 
                
                # get original weight magnitude
                w_abs = self._get_score(m)
                n_wg = len(w_abs)
                self.ranking[name] = []
                for _ in range(n_wg):
                    self.ranking[name].append([])
                self.original_w_mag[name] = m.weight.abs().mean().item()
                kept_wg_L1 = [i for i in range(n_wg) if i not in self.pruned_wg_L1[name]]
                self.original_kept_w_mag[name] = w_abs[kept_wg_L1].mean().item()

    def _pick_pruned_wg(self, w, pr):
        if pr == 0:
            return []
        elif pr > 0:
            w = w.flatten()
            n_pruned = min(math.ceil(pr * w.size(0)), w.size(0) - 1) # do not prune all
            return w.sort()[1][:n_pruned]
        elif pr == -1: # automatically decide lr by each layer itself
            tmp = w.flatten().sort()[0]
            n_not_consider = int(len(tmp) * 0.02)
            w = tmp[n_not_consider:-n_not_consider]

            sorted_w, sorted_index = w.flatten().sort()
            max_gap = 0
            max_index = 0
            for i in range(len(sorted_w) - 1):
                # gap = sorted_w[i+1:].mean() - sorted_w[:i+1].mean()
                gap = sorted_w[i+1] - sorted_w[i]
                if gap > max_gap:
                    max_gap = gap
                    max_index = i
            max_index += n_not_consider
            return sorted_index[:max_index + 1]
        else:
            self.logprint("Wrong pr. Please check.")
            exit(1)
    
    def _update_mag_ratio(self, m, name, w_abs, pruned=None):
        if type(pruned) == type(None):
            pruned = self.pruned_wg[name]
        kept = [i for i in range(len(w_abs)) if i not in pruned]
        ave_mag_pruned = w_abs[pruned].mean()
        ave_mag_kept = w_abs[kept].mean()
        if len(pruned):
            mag_ratio = ave_mag_kept / ave_mag_pruned 
            if name in self.hist_mag_ratio:
                self.hist_mag_ratio[name] = self.hist_mag_ratio[name]* 0.9 + mag_ratio * 0.1
            else:
                self.hist_mag_ratio[name] = mag_ratio
        else:
            mag_ratio = math.inf
            self.hist_mag_ratio[name] = math.inf
        
        # print
        mag_ratio_now_before = ave_mag_kept / self.original_kept_w_mag[name]
        if self.total_iter % self.args.print_interval == 0:
            self.logprint("    mag_ratio %.4f mag_ratio_momentum %.4f" % (mag_ratio, self.hist_mag_ratio[name]))
            self.logprint("    for kept weights, original_kept_w_mag %.6f, now_kept_w_mag %.6f ratio_now_over_original %.4f" % 
                (self.original_kept_w_mag[name], ave_mag_kept, mag_ratio_now_before))
        return mag_ratio_now_before

    def _get_score(self, m):
        shape = m.weight.data.shape
        if self.args.wg == "channel":
            w_abs = m.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=0)
        elif self.args.wg == "filter":
            w_abs = m.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=1)
        elif self.args.wg == "weight":
            w_abs = m.weight.abs().flatten()
        return w_abs

    def _fix_reg(self, m, name):
        if self.pr[name] == 0:
            return True
        if self.args.wg != 'weight':
            self._update_mag_ratio(m, name, self.w_abs[name])

        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] = self.args.reg_upper_limit
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] = self.args.reg_upper_limit
        elif self.args.wg == 'weight':
            self.reg[name][pruned] = self.args.reg_upper_limit

        finish_update_reg = self.total_iter > self.args.fix_reg_interval
        return finish_update_reg

    def _greg_1(self, m, name):
        if self.pr[name] == 0:
            return True
        
        if self.args.wg != 'weight': # weight is too slow
            self._update_mag_ratio(m, name, self.w_abs[name])
        
        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name][pruned] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        if self.args.wg == 'weight': # for weight, do not use the magnitude ratio condition, because 'hist_mag_ratio' is not updated, too costly
            finish_update_reg = False
        else:
            finish_update_reg = True
            for k in self.hist_mag_ratio:
                if self.hist_mag_ratio[k] < self.args.mag_ratio_limit:
                    finish_update_reg = False
        return finish_update_reg or self.reg[name].max() > self.args.reg_upper_limit
        
    def _greg_2(self, m, name):
        layer_index = self.layers[name].layer_index
        w_abs = self.w_abs[name]
        n_wg = len(w_abs)
        pr = self.pr[name]
        if pr == 0:
            self.kept_wg[name] = range(n_wg)
            self.pruned_wg[name] = []
            self.iter_finish_pick[name] = self.total_iter
            return True
        
        if name in self.iter_finish_pick:
            recover_reg = self.args.reg_granularity_recover
            # for pruned weights, push them more
            if self.args.wg == 'channel':
                self.reg[name][:, self.pruned_wg[name]] += self.args.reg_granularity_prune
                self.reg[name][:, self.kept_wg[name]] = recover_reg
            elif self.args.wg == 'filter':
                self.reg[name][self.pruned_wg[name], :] += self.args.reg_granularity_prune
                self.reg[name][self.kept_wg[name], :] = recover_reg
            elif self.args.wg == 'weight':
                self.reg[name][self.pruned_wg[name]] += self.args.reg_granularity_prune
                self.reg[name][self.kept_wg[name]] = recover_reg

            # for kept weights, bring them back
            # 09/22 update: It seems negative reg is a bad idea to bring back magnitude.
            '''
            current_w_mag = w_abs[self.kept_wg[name]].mean()
            recover_reg = (current_w_mag / self.original_kept_w_mag[name] - 1).item() \
                * self.args.weight_decay * self.args.reg_multiplier * 10
            if recover_reg > 0:
                recover_reg = 0
            if self.args.wg == 'channel':
                self.reg[name][:, self.kept_wg[name]] = recover_reg
            elif self.args.wg == 'filter':
                self.reg[name][self.kept_wg[name], :] = recover_reg
            '''
            if self.total_iter % self.args.print_interval == 0:
                self.logprint("    prune stage, push the pruned (reg = %.5f) to zero; for kept weights, reg = %.5f" 
                    % (self.reg[name].max().item(), recover_reg))
            
        else:
            self.reg[name] += self.args.reg_granularity_pick

        # plot w_abs distribution
        if self.total_iter % self.args.plot_interval == 0:
            self._plot_mag_ratio(w_abs, name)
        if self.total_iter % self.args.plot_interval == 0:
            self._log_down_mag_reg(w_abs, name)
        
        # save order
        # if not hasattr(self, 'order_by_L1'):
            # self.order_by_L1 = {}
            # self.wg_preprune = {} # deprecated: dict costs too much memory and seriously slow down training
        # if name in self.order_by_L1:
        #     self.order_by_L1[name] = torch.cat([self.order_by_L1[name], order], dim=0)
        #     self.wg_preprune[name] = torch.cat([self.wg_preprune[name], wg_pruned], dim=0)
        # else:
        #     self.order_by_L1[name] = order
        #     self.wg_preprune[name] = wg_pruned
        if self.args.save_order_log and self.total_iter % self.args.update_reg_interval == 0:
            if not hasattr(self, 'order_log'):
                self.order_log = open('%s/order_log.txt' % self.logger.log_path, 'w+')

            order = w_abs.argsort().argsort()
            n_pruned = min(math.ceil(pr * n_wg), n_wg - 1)
            wg_pruned = w_abs.argsort()[:n_pruned]
            
            order = [str(x.item()) for x in order] # for each wg, its ranking
            wg_pruned = [str(x.item()) for x in wg_pruned]

            logtmp = ' '.join(order)
            logtmp = 'Iter %d Layer#%d %s order_by_L1 %s' % (self.total_iter, layer_index, name, logtmp)
            print(logtmp, file=self.order_log)

            logtmp = ' '.join(wg_pruned)
            logtmp = 'Iter %d Layer#%d %s wg_preprune %s' % (self.total_iter, layer_index, name, logtmp)
            print(logtmp, file=self.order_log, flush=True)
            
        # print to check magnitude ratio
        if self.args.wg != 'weight':
            if name in self.iter_finish_pick:
                self._update_mag_ratio(m, name, w_abs)
            else:
                pruned_wg = self._pick_pruned_wg(w_abs, pr)
                self._update_mag_ratio(m, name, w_abs, pruned=pruned_wg) # just print to check
            
        # check if picking finishes
        finish_pick_cond = self.reg[name].max() >= self.args.reg_upper_limit_pick
        if name not in self.iter_finish_pick and finish_pick_cond:
            self.iter_finish_pick[name] = self.total_iter
            pruned_wg = self._pick_pruned_wg(w_abs, pr)
            kept_wg = [i for i in range(n_wg) if i not in pruned_wg]
            self.kept_wg[name] = kept_wg
            self.pruned_wg[name] = pruned_wg
            picked_wg_in_common = [i for i in pruned_wg if i in self.pruned_wg_L1[name]]
            common_ratio = len(picked_wg_in_common) / len(pruned_wg) if len(pruned_wg) else -1
            n_finish_pick = len(self.iter_finish_pick)
            self.logprint("    [%d] just finished pick (n_finish_pick = %d). %.2f in common chosen by L1 & GReg-2. Iter = %d" % 
                (layer_index, n_finish_pick, common_ratio, self.total_iter))
            
            # re-scale the weights to recover the response magnitude
            # factor = self.original_w_mag[name] / m.weight.abs().mean()
            # m.weight.data.mul_(factor)
            # self.logprint('    rescale weight by %.4f' % factor.item())

            # check if all layers finish picking
            self.all_layer_finish_pick = True
            for k in self.reg:
                if self.pr[k] > 0 and (k not in self.iter_finish_pick):
                    self.all_layer_finish_pick = False
                    break
        
        # save mag_reg_log
        if self.args.save_mag_reg_log and (self.total_iter % self.args.save_interval == 0 or \
                name in self.iter_finish_pick):
            out = pjoin(self.logger.log_path, "%d_mag_reg_log.npy" % layer_index)
            np.save(out, self.mag_reg_log[name])
            if self.all_layer_finish_pick:
                exit(0)
        
        if self.args.__dict__.get('AdaReg_only_picking') or self.args.__dict__.get('AdaReg_revive_kept'):
            finish_update_reg = False
        else:
            cond0 = name in self.iter_finish_pick # finsihed picking
            if self.args.wg == 'weight':
                cond1 = self.reg[name].max() > self.args.reg_upper_limit
            else:
                cond1 = self.hist_mag_ratio[name] >= self.args.mag_ratio_limit \
                    or self.reg[name].max() > self.args.reg_upper_limit
            finish_update_reg = cond0 and cond1
        return finish_update_reg

    def _update_reg(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                cnt_m = self.layers[name].layer_index
                pr = self.pr[name]
                
                if name in self.iter_update_reg_finished.keys():
                    continue

                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("[%d] Update reg for layer '%s'. Pr = %s. Iter = %d" 
                        % (cnt_m, name, pr, self.total_iter))
                
                # get the importance score (L1-norm in this case)
                self.w_abs[name] = self._get_score(m)
                
                # update reg functions, two things: 
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.method == "FixReg":
                    finish_update_reg = self._fix_reg(m, name)
                elif self.args.method == "GReg-1":
                    finish_update_reg = self._greg_1(m, name)
                elif self.args.method == "GReg-2":
                    finish_update_reg = self._greg_2(m, name)
                else:
                    self.logprint("Wrong '--method' argument, please check.")
                    exit(1)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint("==> [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, nn.Conv2d) or isinstance(mm, nn.Linear):
                            if n not in self.iter_update_reg_finished:
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)
                        self._save_model(mark='just_finished_update_reg')
                    
                # after reg is updated, print to check
                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("    reg_status: min = %.5f ave = %.5f max = %.5f" % 
                                (self.reg[name].min(), self.reg[name].mean(), self.reg[name].max()))

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.reg:
                reg = self.reg[name] # [N, C]
                if self.args.wg in ['filter', 'channel']:
                    if reg.shape != m.weight.data.shape:
                        reg = reg.unsqueeze(2).unsqueeze(3) # [N, C, 1, 1]
                elif self.args.wg == 'weight':
                    reg = reg.view_as(m.weight.data) # [N, C, H, W]
                l2_grad = reg * m.weight
                if self.args.block_loss_grad:
                    m.weight.grad = l2_grad
                else:
                    m.weight.grad += l2_grad
    
    def _resume_prune_status(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.model = state['model'].cuda()
        self.model.load_state_dict(state['state_dict'])
        self.optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.args.lr_pick if self.args.__dict__.get('AdaReg_only_picking') else self.args.lr_prune, 
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        self.optimizer.load_state_dict(state['optimizer'])
        self.prune_state = state['prune_state']
        self.total_iter = state['iter']
        self.iter_stabilize_reg = state.get('iter_stabilize_reg', math.inf)
        self.reg = state['reg']
        self.hist_mag_ratio = state['hist_mag_ratio']

    def _save_model(self, acc1=0, acc5=0, mark=''):
        state = {'iter': self.total_iter,
                'prune_state': self.prune_state, # we will resume prune_state
                'arch': self.args.arch,
                'model': self.model,
                'state_dict': self.model.state_dict(),
                'iter_stabilize_reg': self.iter_stabilize_reg,
                'acc1': acc1,
                'acc5': acc5,
                'optimizer': self.optimizer.state_dict(),
                'reg': self.reg,
                'hist_mag_ratio': self.hist_mag_ratio,
                'ExpID': self.logger.ExpID,
        }
        self.save(state, is_best=False, mark=mark)

    def prune(self):
        self.model = self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.args.lr_pick if self.args.__dict__.get('AdaReg_only_picking') else self.args.lr_prune,
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        
        # resume model, optimzer, prune_status
        self.total_iter = -1
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)
            self._get_kept_wg_L1() # get pruned and kept wg from the resumed model
            self.model = self.model.train()
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                        self.args.resume_path, self.total_iter, self.prune_state))

        t1 = time.time()
        acc1 = acc5 = 0
        while True:
            for _, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.total_iter += 1
                total_iter = self.total_iter
                
                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(self.model)
                    self.accprint("Acc1 = %.4f Acc5 = %.4f Iter = %d (before update) [prune_state = %s, method = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state, self.args.method))
                
                # save model (save model before a batch starts)
                if total_iter % self.args.save_interval == 0:
                    self._save_model(acc1, acc5)
                    self.logprint('Periodically save model done. Iter = {}'.format(total_iter))
                    
                if total_iter % self.args.print_interval == 0:
                    self.logprint("")
                    self.logprint("Iter = %d [prune_state = %s, method = %s] " 
                        % (total_iter, self.prune_state, self.args.method) + "-"*40)
                    
                # forward
                self.model.train()
                y_ = self.model(inputs)
                
                if self.prune_state == "update_reg" and total_iter % self.args.update_reg_interval == 0:
                    self._update_reg()
                    
                # normal training forward
                loss = self.criterion(y_, targets)
                self.optimizer.zero_grad()
                loss.backward()
                
                # after backward but before update, apply reg to the grad
                self._apply_reg()
                self.optimizer.step()

                # log print
                if total_iter % self.args.print_interval == 0:
                    w_abs_sum = 0
                    w_num_sum = 0
                    cnt_m = 0
                    for _, m in self.model.named_modules():
                        if isinstance(m, nn.Conv2d):
                            cnt_m += 1
                            w_abs_sum += m.weight.abs().sum()
                            w_num_sum += m.weight.numel()

                    _, predicted = y_.max(1)
                    correct = predicted.eq(targets).sum().item()
                    train_acc = correct / targets.size(0)
                    self.logprint("After optim update, ave_abs_weight: %.10f current_train_loss: %.4f current_train_acc: %.4f" %
                        (w_abs_sum / w_num_sum, loss.item(), train_acc))
                
                # Save heatmap of weights to check the magnitude
                # if total_iter % self.args.plot_interval == 0:
                #     cnt_m = 0
                #     for m in self.modules:
                #         cnt_m += 1
                #         if isinstance(m, nn.Conv2d):
                #             out_path1 = os.path.join(self.logger.logplt_path, "m%d_iter%d_weights_heatmap.jpg" % 
                #                     (cnt_m, total_iter))
                #             out_path2 = os.path.join(self.logger.logplt_path, "m%d_iter%d_reg_heatmap.jpg" % 
                #                     (cnt_m, total_iter))
                #             plot_weights_heatmap(m.weight.mean(dim=[2, 3]), out_path1)
                #             plot_weights_heatmap(self.reg[name], out_path2)
                
                if self.args.__dict__.get('AdaReg_only_picking') and self.all_layer_finish_pick:
                    self.logprint("GReg-2 just finished picking for all layers. Resume original model and switch to GReg-1. Iter = %d" % total_iter)
                    
                    # save picked wg
                    pkl_path = os.path.join(self.logger.log_path, 'picked_wg.pkl')
                    with open(pkl_path, "wb" ) as f:
                        pickle.dump(self.pruned_wg, f)
                    exit(0)
                    
                    # set to GReg-1 method
                    self.model = self.original_model # reload the original model
                    self.optimizer = optim.SGD(self.model.parameters(), 
                                            lr=self.args.lr_prune, 
                                            momentum=self.args.momentum,
                                            weight_decay=self.args.weight_decay)
                    self.args.method = "GReg-1"
                    self.args.AdaReg_only_picking = False # do not get in again
                    # reinit
                    for k in self.reg:
                        self.reg[k] = torch.zeros_like(self.reg[k]).cuda()
                    self.hist_mag_ratio = {}
                
                if self.args.__dict__.get('AdaReg_revive_kept') and self.all_layer_finish_pick:
                    self._prune_and_build_new_model()
                    self.logprint("GReg-2 just finished picking for all layers. Pruned and go to 'finetune'. Iter = %d" % total_iter)
                    return copy.deepcopy(self.model)
                
                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                    self._prune_and_build_new_model() 
                    self.logprint("'stabilize_reg' is done. Pruned, go to 'finetune'. Iter = %d" % total_iter)
                    return copy.deepcopy(self.model)

                if total_iter % self.args.print_interval == 0:
                    t2 = time.time()
                    total_time = t2 - t1
                    self.logprint("speed = %.4f iter/s" % (self.args.print_interval / total_time))
                    t1 = t2

    def _plot_mag_ratio(self, w_abs, name):
        fig, ax = plt.subplots()
        max_ = w_abs.max().item()
        w_abs_normalized = (w_abs / max_).data.cpu().numpy()
        ax.plot(w_abs_normalized)
        ax.set_ylim([0, 1])
        ax.set_xlabel('filter index')
        ax.set_ylabel('relative L1-norm ratio')
        layer_index = self.layers[name].layer_index
        shape = self.layers[name].size
        ax.set_title("layer %d iter %d shape %s\n(max = %s)" 
            % (layer_index, self.total_iter, shape, max_))
        out = pjoin(self.logger.logplt_path, "%d_iter%d_w_abs_dist.jpg" % 
                                (layer_index, self.total_iter))
        fig.savefig(out)
        plt.close(fig)
        np.save(out.replace('.jpg', '.npy'), w_abs_normalized)

    def _log_down_mag_reg(self, w_abs, name):
        step = self.total_iter
        reg = self.reg[name].max().item()
        mag = w_abs.data.cpu().numpy()
        if name not in self.mag_reg_log:
            values = [[step, reg, mag]]
            log = {
                'name': name,
                'layer_index': self.layers[name].layer_index,
                'shape': self.layers[name].size,
                'values': values,
            }
            self.mag_reg_log[name] = log
        else:
            values = self.mag_reg_log[name]['values']
            values.append([step, reg, mag])