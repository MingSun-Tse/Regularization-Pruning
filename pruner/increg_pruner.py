import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import time
import numpy as np
from pruner import Pruner
from utils import plot_weights_heatmap
import math
import matplotlib.pyplot as plt
pjoin = os.path.join

class IncRegPruner(Pruner):
    def __init__(self, model, args, logger, runner):
        super(IncRegPruner, self).__init__(model, args, logger, runner)

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self.hist_mag_ratio = {}
        self.n_update_reg = {}
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.original_w_mag = {}
        self.ranking = {}
        self.pruned_wg_L1 = {}
        self.all_layer_finish_pick = False
        self.w_abs = {}
        if self.args.AdaReg_only_picking:
            self.original_model = copy.deepcopy(self.model)
        
        # prune_init, to determine the pruned weights
        # this will update the 'self.kept_wg' and 'self.pruned_wg' 
        if self.args.method.endswith('Reg'):
            self._get_kept_wg_L1()
            if self.args.method == "AdaReg":
                for k, v in self.pruned_wg.items():
                    self.pruned_wg_L1[k] = v
                self.pruned_wg = {}
                self.kept_wg = {}

        self.prune_state = "update_reg"
        if self.args.method.endswith("Reg"):
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    N, C, H, W = m.weight.data.size()
                    self.reg[name] = torch.zeros(N, C).cuda()
                    self.original_w_mag[name] = m.weight.abs().mean().item()
                    self.ranking[name] = []
                    if self.args.wg == "filter":
                        n_wg = N
                    elif self.args.wg == "channel":
                        n_wg = C
                    for _ in range(n_wg):
                        self.ranking[name].append([])

    def _pick_pruned_wg(self, w, pr):
        if pr == 0:
            return []
        elif pr > 0:
            w = w.flatten()
            return w.sort()[1][:math.ceil(pr * w.size(0))]
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
            self.print("Wrong pr. Please check.")
            exit(1)
    
    def _get_volatility(self, ranking):
        return np.max(ranking[-10:]) - np.min(ranking[-10:])
    
    def _update_mag_ratio(self, m, name, w_abs, pruned=None):
        if type(pruned) == type(None):
            pruned = self.pruned_wg[m]
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
        mag_ratio_now_before = ave_mag_kept / self.original_w_mag[name]
        if self.total_iter % self.args.print_interval == 0:
            self.print("    Mag ratio = %.4f (%.4f)" % (mag_ratio, self.hist_mag_ratio[name]))
            self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % 
                (self.original_w_mag[name], ave_mag_kept, mag_ratio_now_before))

    def _get_score(self, m):
        if self.args.wg == "channel":
            w_abs = m.weight.abs().mean(dim=[0, 2, 3])
        elif self.args.wg == "filter":
            w_abs = m.weight.abs().mean(dim=[1, 2, 3])
        return w_abs

    def _fix_reg(self, m, name):
        if self._get_layer_pr(name) == 0:
            return True
        pruned = self.pruned_wg[m]
        self._update_mag_ratio(m, name, self.w_abs[name])

        if self.args.wg == "channel":
            self.reg[name][:, pruned] = 1e4 * self.args.weight_decay
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] = 1e4 * self.args.weight_decay

        finish_condition = self.total_iter > 10000
        return finish_condition

    def _inc_reg(self, m, name):
        if self._get_layer_pr(name) == 0:
            return True
        pruned = self.pruned_wg[m]
        self._update_mag_ratio(m, name, self.w_abs[name])
        
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.weight_decay
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] += self.args.weight_decay
        
        # when all layers are pushed hard enough, stop
        finish_condition = True
        for k in self.hist_mag_ratio:
            if self.hist_mag_ratio[k] < 1000:
                finish_condition = False
        return finish_condition

    def _ada_reg(self, m, name):
        pr = self._get_layer_pr(name)
        if pr == 0:
            return True
        w_abs = self.w_abs[name]
        n_wg = len(w_abs)
        if 0: # m in self.iter_finish_pick.keys(): # not use for now
            # for pruned weights, push them more 
            if self.args.wg == 'channel':
                self.reg[name][:, self.pruned_wg[m]] += self.args.weight_decay * 10
            elif self.args.wg == 'filter':
                self.reg[name][self.pruned_wg[m], :] += self.args.weight_decay * 10

            # for kept weights, bring them back
            current_w_mag = w_abs[self.kept_wg[m]].mean()
            recover_reg = min((current_w_mag / self.original_w_mag[name] - 1) * self.args.weight_decay * 10, 
                    self.args.weight_decay)
            if self.args.wg == 'channel':
                self.reg[name][:, self.kept_wg[m]] = recover_reg
            elif self.args.wg == 'filter':
                self.reg[name][self.kept_wg[m], :] = recover_reg
        else:
            self.reg[name] += self.args.weight_decay

#                         # check ranking volatility
#                         current_ranking = w_abs.sort()[1] # ranking of different weight groups
#                         logtmp = "    Rank_volatility: "
#                         v = []
#                         cnt_reward = 0
#                         for i in range(n_wg):
#                             chl = current_ranking[i]
#                             self.ranking[m][chl].append(i)
#                             volatility = self._get_volatility(self.ranking[m][chl])
#                             logtmp += "%d " % volatility
#                             v.append(volatility)

# #                                 # Reg reward
# #                                 # if chl is good now and quite stable, it signs that this chl probably will be kept finally,
# #                                 # so reward it.
# #                                 if len(self.ranking[m][chl]) > 10:
# #                                     if i >= int(self.args.prune_ratio * C) and volatility <= 0.02 * C:
# #                                         cnt_reward += 1
# #                                         self.reg[name][:, chl] -= self.args.weight_decay * 4
# #                                         self.reg[name][:, chl] = torch.max(self.reg[name][:, chl], torch.zeros(N).cuda())

#                         # print and plot
#                         if self.total_iter % self.args.print_interval == 0:
#                             self.print(logtmp)
#                             self.print("    Reward_ratio = %.4f" % (cnt_reward / C))

#                             # plot
#                             # if self.total_iter % (self.args.print_interval * 10) == 0:
#                             #     fig, ax = plt.subplots()
#                             #     ax.plot(v)
#                             #     ax.set_ylim([0, 100])
#                             #     out = os.path.join(self.logger.logplt_path, "%d_iter%d_ranking.jpg" % 
#                             #                           (cnt_m, self.total_iter))
#                             #     fig.savefig(out)
#                             #     plt.close(fig)

        # plot w_abs distribution
        if self.total_iter % self.args.plot_interval == 0:
            fig, ax = plt.subplots()
            sorted_w_abs = w_abs.sort()[0].data.cpu().numpy()
            max = sorted_w_abs[-1]
            sorted_w_abs /= max # normalize
            ax.plot(sorted_w_abs)
            ax.set_ylim([0, 1])
            ax.set_title("max = %s" % max)
            layer_index = self.layers[name].layer_index
            out = pjoin(self.logger.logplt_path, "%d_iter%d_w_abs_dist.jpg" % 
                                    (layer_index, self.total_iter))
            fig.savefig(out)
            plt.close(fig)

        # print to check magnitude ratio
        if self.total_iter % self.args.pick_pruned_interval == 0:
            pruned_wg = self._pick_pruned_wg(w_abs, pr)
            # self.print("    Pruned_wg (pr=%.4f): " % (len(pruned_wg) / n_wg))
            self._update_mag_ratio(m, name, w_abs, pruned=pruned_wg)
            
        # check if picking finishes
        finish_pick_cond = self.reg[name].max() >= self.args.reg_upper_limit_pick
        if name not in self.iter_finish_pick and finish_pick_cond:
            self.iter_finish_pick[name] = self.total_iter
            pruned_wg = self._pick_pruned_wg(w_abs, pr)
            kept_wg = [i for i in range(n_wg) if i not in pruned_wg]
            self.kept_wg[m] = kept_wg
            self.pruned_wg[m] = pruned_wg
            picked_wg_in_common = [i for i in pruned_wg if i in self.pruned_wg_L1[m]]
            common_ratio = len(picked_wg_in_common) / len(pruned_wg) if len(pruned_wg) else -1
            layer_index = self.layers[name].layer_index
            self.print("    ! [%d] Just finished picking the pruned. Iter = %d" % (layer_index, self.total_iter))
            self.print("    %.2f weight groups chosen by L1 and AdaReg in common" % common_ratio)

            # check if all layer finishes picking
            self.all_layer_finish_pick = True
            for k in self.reg.keys():
                if self._get_layer_pr(k) > 0:
                    if k not in self.iter_finish_pick.keys():
                        self.all_layer_finish_pick = False
                        break
        
        finish_condition = self.hist_mag_ratio[name] > 1000 and self.mag_ratio_now_before > 0.95
        return finish_condition

    def _update_reg(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                cnt_m = self.layers[name].layer_index
                
                if m in self.iter_update_reg_finished.keys():
                    continue

                if self.total_iter % self.args.print_interval == 0:
                    self.print("[%d] Update reg for layer '%s'. Iter = %d. Method = %s" 
                        % (cnt_m, name, self.total_iter, self.args.method))
                
                # get the importance score (L1-norm in this case)
                self.w_abs[name] = self._get_score(m)
                
                # update reg functions, two things: 
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.method == "FixReg":
                    finish_condition = self._fix_reg(m, name)
                elif self.args.method == "IncReg":
                    finish_condition = self._inc_reg(m, name)
                elif self.args.method == "AdaReg":
                    finish_condition = self._ada_reg(m, name)
                else:
                    self.print("Wrong 'method'. Please check.")
                    exit(1)

                # check prune state
                if finish_condition:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[m] = self.total_iter
                    self.print("    ! [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stabilize_reg"
                    for mm in self.model.modules():
                        if isinstance(mm, nn.Conv2d):
                            if mm not in self.iter_update_reg_finished.keys():
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                        self.print("    ! All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)
                    
                    # not used for now
                    # if self.args.method == "AdaReg":
                    #     if self.args.wg == 'channel':
                    #         self.reg[name][:, self.kept_wg[m]] = 0 
                    #     elif self.args.wg == 'filter':
                    #         self.reg[name][self.kept_wg[m], :] = 0

                # after reg is updated, print to check
                if self.total_iter % self.args.print_interval == 0:
                    self.print("    Reg status: min = %.5f ave = %.5f max = %.5f" % 
                                (self.reg[name].min(), self.reg[name].mean(), self.reg[name].max()))

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.reg:
                reg = self.reg[name] # [N, C]
                reg = reg.unsqueeze(2).unsqueeze(3) # [N, C, 1, 1]
                l2_grad = reg * m.weight
                m.weight.grad += l2_grad

    def prune(self):
        self.model = self.model.train()
        optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.args.lr_prune, 
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        epoch = -1
        t1 = time.time()
        while True:
            epoch += 1
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                total_iter = epoch * len(self.train_loader) + batch_idx
                self.total_iter = total_iter
                
                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5 = self.test(self.model)
                    self.print("Acc1 = %.4f Acc5 = %.4f Iter = %d (before update) [prune_state = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state))
                    
                if total_iter % self.args.print_interval == 0:
                    self.print("")
                    self.print("Iter = %d" % total_iter + " [prune_state = %s] " % self.prune_state + "-"*40)
                    
                # forward
                y_ = self.model(inputs)
                
                if self.prune_state == "update_reg" and total_iter % self.args.update_reg_interval == 0:
                    # if self.args.method == "OptReg":
                    #     # estimate K-FAC fisher as Hessian
                    #     softmax_y = y_.softmax(dim=1)
                    #     if softmax_y.min() < 0 or math.isnan(softmax_y.mean()):
                    #         print(softmax_y)
                    #         print(y_)
                    #         exit(1)
                    #     sampled_y = torch.multinomial(softmax_y, 1).squeeze()
                    #     loss_sample = self.criterion(y_, sampled_y)
                    #     loss_sample.backward(retain_graph=True)
                    
                    # update reg
                    self._update_reg()
                    
                # normal training forward
                loss = self.criterion(y_, targets)
                optimizer.zero_grad()
                loss.backward()
                
                # after backward but before update, apply reg to the grad
                self._apply_reg()
                optimizer.step()

                # log print
                if total_iter % self.args.print_interval == 0:
                    w_abs_sum = 0
                    w_num_sum = 0
                    cnt_m = 0
                    for name, m in self.model.named_modules():
                        if isinstance(m, nn.Conv2d):
                            cnt_m += 1
                            w_abs_sum += m.weight.abs().sum()
                            w_num_sum += m.weight.numel()

                            # w magnitude print
                            # w_abs_mean = m.weight.abs().mean(dim=[2, 3]) # [N, C]
                            # logtmp1 = ""
                            # for x in torch.diag(w_abs_mean)[:20]:
                            #     logtmp1 += "{:8.6f} ".format(x.item())  
                            # logtmp2 = ""
                            # for x in torch.diag(self.reg[name])[:20]:
                            #     logtmp2 += "{:8.6f} ".format(x.item())
                            # self.print("{:2d}: {}".format(cnt_m, logtmp2))
                            # self.print("  : {}".format(logtmp1))
                            # self.print("")

                    _, predicted = y_.max(1)
                    correct = predicted.eq(targets).sum().item()
                    train_acc = correct / targets.size(0)
                    self.print("After optim update, ave_abs_weight: %.10f current_train_loss: %.4f current_train_acc: %.4f" %
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
                
                if self.args.AdaReg_only_picking and self.all_layer_finish_pick:
                    self.print("AdaReg just finished picking pruned wg for all layers. Iter = %d" % total_iter)
                    # update key
                    for m_old, m_new in zip(self.model.modules(), self.original_model.modules()):
                        if m_old in self.kept_wg.keys():
                            self.kept_wg[m_new] = self.kept_wg[m_old]
                            self.pruned_wg[m_new] = self.pruned_wg[m_old]
                            self.kept_wg.pop(m_old)
                            self.pruned_wg.pop(m_old)
                    self.model = self.original_model # reload the original model
                    self.args.method = "IncReg"
                    self.args.AdaReg_only_picking = False # do not get in again
                    # reinit
                    for k in self.reg:
                        self.reg[k] = torch.zeros_like(self.reg[k]).cuda()
                    self.hist_mag_ratio = {}
                
                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                    self.print("'stabilize_reg' is done. Now prune. Iter = %d" % total_iter)
                    self._prune_and_build_new_model() 
                    self.print("Prune is done, go to 'finetune'")
                    return copy.deepcopy(self.model) # the only out of this loop

                if total_iter % self.args.print_interval == 0:
                    t2 = time.time()
                    total_time = t2 - t1
                    self.print("speed = %.2f iter/s" % (self.args.print_interval / total_time))
                    t1 = t2