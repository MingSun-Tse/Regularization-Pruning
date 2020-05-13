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

        # IncReg related variables
        self.reg = {}
        self.delta_reg = {}
        self.hist_mag_ratio = {}
        self.n_update_reg = {}
        self.iter_update_reg_finished = {}
        self.iter_pick_pruned_finished = {}
        self.original_w_mag = {}
        self.ranking = {}
        self.pruned_wg_L1 = {}
        self.all_layer_finish_picking = False
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
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    N, C, H, W = m.weight.data.size()
                    self.reg[m] = torch.zeros(N, C).cuda()
                    self.original_w_mag[m] = m.weight.abs().mean()
                    self.ranking[m] = []
                    if self.args.wg == "filter":
                        n_wg = N
                    elif self.args.wg == "channel":
                        n_wg = C
                    for _ in range(n_wg):
                        self.ranking[m].append([])

                
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
    
    def _get_mag_ratio(self, m, pruned):
        N, C, H, W = m.weight.size()
        if self.args.wg == "channel":
            w_abs = m.weight.abs().mean(dim=[0, 2, 3])
            n_wg = C
        elif self.args.wg == "filter":
            w_abs = m.weight.abs().mean(dim=[1, 2, 3])
            n_wg = N
        
        ave_mag_pruned = w_abs[pruned].mean()
        ave_mag_kept = (w_abs.sum() - w_abs[pruned].sum()) / (n_wg - len(pruned))
        if len(pruned):
            mag_ratio = ave_mag_kept / ave_mag_pruned 
            if m in self.hist_mag_ratio.keys():
                self.hist_mag_ratio[m] = self.hist_mag_ratio[m]* 0.9 + mag_ratio * 0.1
            else:
                self.hist_mag_ratio[m] = mag_ratio
        else:
            mag_ratio = 123456789
            self.hist_mag_ratio[m] = 123456789
        return mag_ratio, self.hist_mag_ratio[m]

    def _update_reg(self):
        cnt_m = 0
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                cnt_m += 1
                pr = self._get_layer_pr(name)
                
                if m in self.iter_update_reg_finished.keys():
                    continue

                if self.total_iter % self.args.print_interval == 0:
                    self.print("[%d] Getting delta reg for module '%s' (iter = %d)" % (cnt_m, m, self.total_iter))
                    
                N, C, H, W = m.weight.size()
                if self.args.wg == "channel":
                    w_abs = m.weight.abs().mean(dim=[0, 2, 3])
                    n_wg = C
                elif self.args.wg == "filter":
                    w_abs = m.weight.abs().mean(dim=[1, 2, 3])
                    n_wg = N

                if self.args.method == "FixReg":
                    pruned = self.pruned_wg[m]
                    kept = self.kept_wg[m]

                    if self.args.wg == "channel":
                        self.reg[m][:, pruned] = 1e4 * self.args.weight_decay
                    elif self.args.wg == "filter":
                        self.reg[m][pruned, :] = 1e4 * self.args.weight_decay

                    mag_ratio, hist_mag_ratio = self._get_mag_ratio(m, pruned)
                    mag_ratio_now_before = w_abs[kept].mean() / self.original_w_mag[m]
                    if self.total_iter % self.args.print_interval == 0:
                        self.print("    Mag ratio = %.2f (%.2f)" % (mag_ratio, hist_mag_ratio))
                        self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % \
                            (self.original_w_mag[m].item(), w_abs[kept].mean().item(), mag_ratio_now_before.item()))

                    # determine if it is time to finish 'update_reg'. When mag ratio is stable.
                    finish_condition = self.total_iter > 10000 and \
                        (mag_ratio - hist_mag_ratio).abs() / hist_mag_ratio < 0.001

                elif self.args.method == "IncReg":
                    pruned = self.pruned_wg[m]
                    kept = self.kept_wg[m]
                    
                    if self.args.wg == "channel":
                        self.reg[m][:, pruned] += self.args.weight_decay
                    elif self.args.wg == "filter":
                        self.reg[m][pruned, :] += self.args.weight_decay

                    mag_ratio, hist_mag_ratio = self._get_mag_ratio(m, pruned)
                    mag_ratio_now_before = w_abs[kept].mean() / self.original_w_mag[m]
                    if self.total_iter % self.args.print_interval == 0:
                        self.print("    Mag ratio = %.2f (%.2f)" % (mag_ratio, hist_mag_ratio))
                        self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % \
                            (self.original_w_mag[m].item(), w_abs[kept].mean().item(), mag_ratio_now_before.item()))
                        
                    # determine if it is time to finish 'update_reg'
                    finish_condition = self.reg[m].max() >= self.args.reg_upper_limit and hist_mag_ratio > 1000

                elif self.args.method == "AdaReg":
                    if m in self.iter_pick_pruned_finished.keys():
                        # for pruned weights, push them more 
                        if self.args.wg == 'channel':
                            self.reg[m][:, self.pruned_wg[m]] += self.args.weight_decay * 10
                        elif self.args.wg == 'filter':
                            self.reg[m][self.pruned_wg[m], :] += self.args.weight_decay * 10

                        # for kept weights, bring them back
                        current_w_mag = w_abs[self.kept_wg[m]].mean()
                        recover_reg = min((current_w_mag / self.original_w_mag[m] - 1) * self.args.weight_decay * 10, 
                                self.args.weight_decay)
                        if self.args.wg == 'channel':
                            self.reg[m][:, self.kept_wg[m]] = recover_reg
                        elif self.args.wg == 'filter':
                            self.reg[m][self.kept_wg[m], :] = recover_reg
                    else:
                        self.reg[m] += self.args.weight_decay

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
# #                                         self.reg[m][:, chl] -= self.args.weight_decay * 4
# #                                         self.reg[m][:, chl] = torch.max(self.reg[m][:, chl], torch.zeros(N).cuda())

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
                        sorted_w_abs /= sorted_w_abs[-1] # normalize
                        ax.plot(sorted_w_abs)
                        ax.set_ylim([0, 1])
                        out = pjoin(self.logger.logplt_path, "%d_iter%d_w_abs_dist.jpg" % 
                                                (cnt_m, self.total_iter))
                        fig.savefig(out)
                        plt.close(fig)

                    # print to check magnitude ratio
                    if self.total_iter % self.args.print_interval == 0:
                        pruned_wg = self._pick_pruned_wg(w_abs, pr)
                        kept_wg = [i for i in range(n_wg) if i not in pruned_wg]
                        mag_ratio, hist_mag_ratio = self._get_mag_ratio(m, pruned_wg)
                        self.mag_ratio_now_before = w_abs[kept_wg].mean() / self.original_w_mag[m]
                        
                        logtmp1 = "    Pruned_wg (pr=%.4f): " % (len(pruned_wg) / n_wg)
                        # for wg in pruned_wg:
                        #     logtmp1 += "%3d " % wg
                        self.print(logtmp1 + "[%d]" % cnt_m)
                        self.print("    Mag ratio = %.2f (%.2f) [%d]" % (mag_ratio, hist_mag_ratio, cnt_m))
                        self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % \
                            (self.original_w_mag[m].item(), w_abs[kept_wg].mean().item(), self.mag_ratio_now_before.item()))

                    # check if picking finishes
                    if m not in self.iter_pick_pruned_finished.keys() and \
                            (self.hist_mag_ratio[m] > self.args.mag_ratio_limit or self.reg[m].max() > 0.2):
                        self.iter_pick_pruned_finished[m] = self.total_iter
                        pruned_wg = self._pick_pruned_wg(w_abs, pr)
                        kept_wg = [i for i in range(n_wg) if i not in pruned_wg]
                        self.kept_wg[m] = kept_wg
                        self.pruned_wg[m] = pruned_wg
                        picked_wg_in_common = [i for i in pruned_wg if i in self.pruned_wg_L1[m]]
                        common_ratio = len(picked_wg_in_common) / len(pruned_wg) if len(pruned_wg) else -1
                        self.print("    Just finish picking the pruned. [%d]. Iter = %d" % (cnt_m, self.total_iter))
                        self.print("    %.2f weight groups chosen by L1 and AdaReg in common" % common_ratio)

                        # check if all layer finishes picking
                        self.all_layer_finish_picking = True
                        for mm in self.model.modules():
                            if isinstance(mm, nn.Conv2d):
                                if mm not in self.iter_pick_pruned_finished.keys():
                                    self.all_layer_finish_picking = False
                                    break
                    
                    finish_condition = self.hist_mag_ratio[m] > 1000 and self.mag_ratio_now_before > 0.95

                elif self.args.method == "OptReg":
                    delta_reg = self._get_delta_reg(m)
                    self.reg[m] = torch.max(self.reg[m] + delta_reg, torch.zeros_like(delta_reg)) # no negative reg

                else:
                    self.print("Wrong 'method'. Please check.")
                    exit(1)

                # log print
                if self.total_iter % self.args.print_interval == 0:
                    self.print("    Reg status: min = %.5f, ave = %.5f, max = %.5f" % 
                                (self.reg[m].min(), self.reg[m].mean(), self.reg[m].max()))

                # check prune state                    
                if pr == 0 or finish_condition:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[m] = self.total_iter
                    self.print("    ! Just finish state 'update_reg'. [%d]. Iter = %d" % (cnt_m, self.total_iter))
                    
                    if self.args.method == "AdaReg":
                        if pr == 0:
                            self.reg[m] = torch.zeros_like(self.reg[m]).cuda()
                        else:
                            if self.args.wg == 'channel':
                                self.reg[m][:, self.kept_wg[m]] = 0 
                            elif self.args.wg == 'filter':
                                self.reg[m][self.kept_wg[m], :] = 0

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stabilize_reg"
                    for mm in self.model.modules():
                        if isinstance(mm, nn.Conv2d):
                            if mm not in self.iter_update_reg_finished.keys():
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                        self.print("    ! All layers have finished state 'update_reg', go to next state 'stabilize_reg'")

    def _get_delta_reg(self, m):
        pass
        return None

    def _apply_reg(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                reg = self.reg[m] # [n_filter, n_channel]
                reg = reg.unsqueeze(2).unsqueeze(3) # [n_filter, n_channel, 1, 1]
                L2_grad = reg * m.weight
                m.weight.grad += L2_grad

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
                
                # Test
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
                    if self.args.method == "OptReg":
                        # estimate K-FAC fisher as Hessian
                        softmax_y = y_.softmax(dim=1)
                        if softmax_y.min() < 0 or math.isnan(softmax_y.mean()):
                            print(softmax_y)
                            print(y_)
                            exit(1)
                        sampled_y = torch.multinomial(softmax_y, 1).squeeze()
                        loss_sample = self.criterion(y_, sampled_y)
                        loss_sample.backward(retain_graph=True)
                    
                    # update reg
                    self._update_reg()
                    
                # Normal training forward
                loss = self.criterion(y_, targets)
                optimizer.zero_grad()
                loss.backward()
                
                # after backward but before update, apply reg to the gradients
                self._apply_reg()
                optimizer.step()

                # Log print 
                _, predicted = y_.max(1)
                correct = predicted.eq(targets).sum().item()
                train_acc = correct / targets.size(0)
                if total_iter % self.args.print_interval == 0:
                    w_abs_sum = 0
                    w_num_sum = 0
                    cnt_m = 0
                    for m in self.model.modules():
                        if isinstance(m, nn.Conv2d):
                            cnt_m += 1
                            w_abs_sum += m.weight.abs().sum()
                            w_num_sum += m.weight.numel()

                            # print
                            w_abs_mean = m.weight.abs().mean(dim=[2, 3]) # [N, C]
                            logtmp1 = ""
                            for x in torch.diag(w_abs_mean)[:20]:
                                logtmp1 += "{:8.6f} ".format(x.item())  
                            logtmp2 = ""
                            for x in torch.diag(self.reg[m])[:20]:
                                logtmp2 += "{:8.6f} ".format(x.item())
                            # self.print("{:2d}: {}".format(cnt_m, logtmp2))
                            # self.print("  : {}".format(logtmp1))
                            # self.print("")
                        
                    self.print("After optim update, ave_abs_weight: %.10f current_train_loss: %.4f current_train_acc: %.4f" %
                        ((w_abs_sum / w_num_sum).item(), loss.item(), train_acc))
                
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
                #             plot_weights_heatmap(self.reg[m], out_path2)
                            
                # Change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg > self.args.stabilize_reg_interval:
                    self.print("State 'stabilize_reg' is done. Now prune at iter %d" % total_iter)
                    self._prune_and_build_new_model() 
                    self.print("Prune is done, go to next state 'finetune'")
                    return copy.deepcopy(self.model)
                
                if self.args.AdaReg_only_picking and self.all_layer_finish_picking:
                    self.print("AdaReg finishes picking pruned channels. Model pruned. Go to 'finetune'")
                    
                    # update key
                    for m_old, m_new in zip(self.model.modules(), self.original_model.modules()):
                        if m_old in self.kept_wg.keys():
                            self.kept_wg[m_new] = self.kept_wg[m_old]
                            self.pruned_wg[m_new] = self.pruned_wg[m_old]
                            self.kept_wg.pop(m_old)
                            self.pruned_wg.pop(m_old)
                    self.model = self.original_model # reload the original model
                    self._prune_and_build_new_model()
                    return copy.deepcopy(self.model)
                
                if total_iter % self.args.print_interval == 0:
                    t2 = time.time()
                    total_time = t2 - t1
                    self.print("speed = %.2f iter/s" % (self.args.print_interval / total_time))
                    t1 = t2