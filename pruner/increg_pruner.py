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
        self.pruned_chl = {}
        self.pruned_chl_L1 = {}
        self.kept_chl = {}
        self.all_layer_finish_picking = False
        if self.args.AdaReg_only_picking:
            self.original_model = copy.deepcopy(self.model)
        
        # init
        if self.args.method in ["FixReg", "IncReg"]:
            self.args.update_interval = 1
            if self.args.arch.startswith("resnet"):
                self._get_kept_chl_L1_resnet(self.args.prune_ratio)
            elif self.args.arch.startswith("alexnet") or self.args.arch.startswith("vgg"):
                self._get_kept_chl_L1(self.args.prune_ratio)

        self.prune_state = "update_reg"
        cnt_m = 0
        for m in self.model.modules():
            cnt_m += 1
            if isinstance(m, nn.Conv2d):
                N, C, H, W = m.weight.data.size()
                self.reg[m] = torch.zeros(N, C).cuda()
                self.original_w_mag[m] = m.weight.abs().mean()
                self.ranking[m] = []
                for _ in range(C):
                    self.ranking[m].append([])
                
    def _get_pruned_channel(self, w, pr=None):
        if pr:
            w = w.flatten()
            return w.sort()[1][:int(pr * w.size(0))]
        else:
            sorted_w, sorted_index = w.flatten().sort()
            max_ratio = 0
            max_index = 0
            for i in range(len(sorted_w) - 1):
                r = sorted_w[i+1:].mean() - sorted_w[:i+1].mean()
                if r > max_ratio:
                    max_ratio = r
                    max_index = i
            return sorted_index[:max_index+1]
    
    def _get_volatility(self, ranking):
        return np.max(ranking[-10:]) - np.min(ranking[-10:])
    
    def _get_mag_ratio(self, m, pruned_chl):
        w_abs = m.weight.abs().mean(dim=[0,2,3])
        N, C, H, W = m.weight.size()
        ave_mag_pruned = w_abs[pruned_chl].mean()
        ave_mag_kept = (w_abs.sum() - w_abs[pruned_chl].sum()) / (C - len(pruned_chl))
        mag_ratio = ave_mag_kept / ave_mag_pruned
        if m in self.hist_mag_ratio.keys():
            self.hist_mag_ratio[m] = self.hist_mag_ratio[m]* 0.9 + mag_ratio * 0.1
        else:
            self.hist_mag_ratio[m] = mag_ratio
        return mag_ratio

    def _update_reg(self):
        cnt_m = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                cnt_m += 1
                
                if self.total_iter % self.args.print_interval == 0:
                    self.print("[%d] Getting delta reg for module '%s' (iter = %d)" % (cnt_m, m, self.total_iter))
                    
                if m in self.iter_update_reg_finished.keys():
                    continue
                    
                N, C, H, W = m.weight.size()
                w_abs = m.weight.abs().mean(dim=[0, 2, 3])
                if self.args.method == "FixReg":
                    self.reg[m][:, self.pruned_chl[m]] = 1e4 * self.args.weight_decay

                    mag_ratio = self._get_mag_ratio(m, self.pruned_chl[m])
                    mag_ratio_now_before = w_abs[self.kept_chl[m]].mean() / self.original_w_mag[m]
                    if self.total_iter % self.args.print_interval == 0:
                        self.print("    Mag ratio = %.2f (%.2f)" % (mag_ratio, self.hist_mag_ratio[m]))
                        self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % \
                            (self.original_w_mag[m].item(), w_abs[self.kept_chl[m]].mean().item(), mag_ratio_now_before.item()))

                    # determine if it is time to finish 'update_reg'. When mag ratio is stable.
                    finish_condition = self.total_iter > 10000 and \
                        (mag_ratio - self.hist_mag_ratio[m]).abs() / self.hist_mag_ratio[m] < 0.001

                elif self.args.method == "IncReg":
                    if self.reg[m].max() < 1e4 * self.args.weight_decay:
                        self.reg[m][:, self.pruned_chl[m]] += self.args.weight_decay

                    mag_ratio = self._get_mag_ratio(m, self.pruned_chl[m])
                    mag_ratio_now_before = w_abs[self.kept_chl[m]].mean() / self.original_w_mag[m]
                    if self.total_iter % self.args.print_interval == 0:
                        self.print("    Mag ratio = %.2f (%.2f)" % (mag_ratio, self.hist_mag_ratio[m]))
                        self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % \
                            (self.original_w_mag[m].item(), w_abs[self.kept_chl[m]].mean().item(), mag_ratio_now_before.item()))
                        
                    # determine if it is time to finish 'update_reg'. When mag ratio is stable.
                    finish_condition = self.total_iter > 10000 and \
                        (mag_ratio - self.hist_mag_ratio[m]).abs() / self.hist_mag_ratio[m] < 0.001

                elif self.args.method == "AdaReg":
                    if m in self.iter_pick_pruned_finished.keys():
                        # for pruned weights, push them more 
                        self.reg[m][:, self.pruned_chl[m]] += self.args.weight_decay * 10

                        # for kept weights, bring them back
                        current_w_mag = w_abs[self.kept_chl[m]].mean()
                        self.reg[m][:, self.kept_chl[m]] = min((current_w_mag / self.original_w_mag[m] - 1) * self.args.weight_decay * 10, self.args.weight_decay)                        
                    else:
                        self.reg[m] += self.args.weight_decay * 2
                        self.w[m] = m.weight.clone()

                        # check ranking volatility
                        current_ranking = w_abs.sort()[1] # ranking of different weight groups
                        logtmp = "    Rank_volatility: "
                        v = []
                        cnt_reward = 0
                        for i in range(C):
                            chl = current_ranking[i]
                            self.ranking[m][chl].append(i)
                            volatility = self._get_volatility(self.ranking[m][chl])
                            logtmp += "%d " % volatility
                            v.append(volatility)

#                                 # Reg reward
#                                 # if chl is good now and quite stable, it signs that this chl probably will be kept finally,
#                                 # so reward it.
#                                 if len(self.ranking[m][chl]) > 10:
#                                     if i >= int(self.args.prune_ratio * C) and volatility <= 0.02 * C:
#                                         cnt_reward += 1
#                                         self.reg[m][:, chl] -= self.args.weight_decay * 4
#                                         self.reg[m][:, chl] = torch.max(self.reg[m][:, chl], torch.zeros(N).cuda())

                        # print and plot
                        if self.total_iter % self.args.print_interval == 0:
                            self.print(logtmp)
                            self.print("    Reward_ratio = %.4f" % (cnt_reward / C))

                            # plot
                            if self.total_iter % (self.args.print_interval * 10) == 0:
                                fig, ax = plt.subplots()
                                ax.plot(v)
                                ax.set_ylim([0, 100])
                                out = os.path.join(self.logger_my.logplt_path, "%d_iter%d_ranking.jpg" % 
                                                      (cnt_m, self.total_iter))
                                fig.savefig(out)
                                plt.close(fig)


                    # check magnitude ratio
                    pruned_chl = self._get_pruned_channel(w_abs, self.args.prune_ratio) # current pruned chl
                    kept_chl = [i for i in range(C) if i not in pruned_chl]
                    mag_ratio = self._get_mag_ratio(m, pruned_chl)
                    mag_ratio_now_before = w_abs[kept_chl].mean() / self.original_w_mag[m]

                    # print
                    if self.total_iter % self.args.print_interval == 0:
                        logtmp1 = "    Pruned_chl (pr=%.4f): " % (len(pruned_chl) / C)
                        for c in pruned_chl:
                            logtmp1 += "%3d " % c
                        self.print(logtmp1 + "[%d]" % cnt_m)
                        self.print("    Mag ratio = %.2f (%.2f) [%d]" % (mag_ratio, self.hist_mag_ratio[m], cnt_m))
                        self.print("    For kept weights, original mag: %.6f, now: %.6f (%.4f)" % \
                            (self.original_w_mag[m].item(), w_abs[kept_chl].mean().item(), mag_ratio_now_before.item()))

                    # check if the picking finishes
                    if m not in self.iter_pick_pruned_finished.keys() and \
                            (self.hist_mag_ratio[m] > self.args.mag_ratio_limit or self.reg[m].max() > 0.2):
                        self.iter_pick_pruned_finished[m] = self.total_iter
                        self.kept_chl[m] = kept_chl
                        self.pruned_chl[m] = pruned_chl
                        picked_chl_in_common = [i for i in pruned_chl if i in self.pruned_chl_L1[m]]
                        common_ratio = len(picked_chl_in_common) / len(pruned_chl)
                        self.print("    Just finish picking the pruned. [%d]. Iter = %d" % (cnt_m, self.total_iter))
                        self.print("    %.2f channels chosen by L1 and AdaReg in common" % common_ratio)

                        # check if all layer finishes picking channels to prune
                        self.all_layer_finish_picking = True
                        for mm in self.model.modules():
                            if isinstance(mm, nn.Conv2d):
                                if mm not in self.iter_pick_pruned_finished.keys():
                                    self.all_layer_finish_picking = False
                                    break
                    
                    finish_condition = self.hist_mag_ratio[m] > 1000 and mag_ratio_now_before > 0.95

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
                if len(self.pruned_chl[m]) == 0 or finish_condition:
                    # after 'update_reg' stage, keep the reg to stablize weight magnitude
                    self.iter_update_reg_finished[m] = self.total_iter
                    self.print("    ! Just finish state 'update_reg'. [%d]. Iter = %d" % (cnt_m, self.total_iter))
                    
                    if self.args.method == "AdaReg":
                        self.reg[m][:, self.kept_chl[m]] = self.args.weight_decay # set back to normal weight decay

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stablize_reg"
                    for mm in self.model.modules():
                        if isinstance(mm, nn.Conv2d):
                            if mm not in self.iter_update_reg_finished.keys():
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stablize_reg":
                        self.iter_stablize_reg = self.total_iter
                        self.print("    ! All layers have finished state 'update_reg', go to next state 'stablize_reg'")

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
                                lr=self.args.lr_pr, 
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        epoch = -1
        while True:
            epoch += 1

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                t1 = time.time()
                inputs, targets = inputs.cuda(), targets.cuda()
                total_iter = epoch * len(self.train_loader) + batch_idx
                self.total_iter = total_iter
                
                # Test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5 = self.test(self.model)
                    self.print("Acc1 = %.4f, Acc5 = %.4f. Total_iter = %d (before update)" % (acc1, acc5, total_iter))
                    
                if total_iter % self.args.print_interval == 0:
                    self.print("")
                    self.print("Total iter = %d" % total_iter + " [prune_state = '%s'] " % self.prune_state + "-"*40)
                    
                # forward
                y_ = self.model(inputs)
                
                if self.prune_state == "update_reg" and total_iter % self.args.update_interval == 0:
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
                if self.args.plot_weights_heatmap and total_iter % 1000 == 0:
                    cnt_m = 0
                    for m in self.modules:
                        cnt_m += 1
                        if isinstance(m, nn.Conv2d):
                            out_path1 = os.path.join(self.logger_my.logplt_path, "m%d_iter%d_weights_heatmap.jpg" % 
                                    (cnt_m, total_iter))
                            out_path2 = os.path.join(self.logger_my.logplt_path, "m%d_iter%d_reg_heatmap.jpg" % 
                                    (cnt_m, total_iter))
                            plot_weights_heatmap(m.weight.mean(dim=[2, 3]), out_path1)
                            plot_weights_heatmap(self.reg[m], out_path2)
                            
                # Change prune state
                if self.prune_state == "stablize_reg" and total_iter - self.iter_stablize_reg > self.args.stablize_interval:
                    self.print("State 'stablize_reg' is done. Now prune at iter %d" % total_iter)
                    self._prune_and_build_new_model() 
                    self.print("Prune is done, go to next state 'finetune'")
                    return self.model
                
                if self.args.AdaReg_only_picking and self.all_layer_finish_picking:
                    self.print("AdaReg finishes picking pruned channels. Model pruned. Go to 'finetune'")
                    
                    # update key
                    for m_old, m_new in zip(self.model.modules(), self.original_model.modules()):
                        if m_old in self.kept_chl.keys():
                            self.kept_chl[m_new] = self.kept_chl[m_old]
                            self.pruned_chl[m_new] = self.pruned_chl[m_old]
                            self.kept_chl.pop(m_old)
                            self.pruned_chl.pop(m_old)
                    self.model = self.original_model # reload the original model
                    self._prune_and_build_new_model()
                    return self.model
                
                # t2 = time.time()
                # total_time = t2 - t1
                # if total_iter % 10 == 0:
                #     self.print("speed = %.2f iter/s" % (1 / total_time))