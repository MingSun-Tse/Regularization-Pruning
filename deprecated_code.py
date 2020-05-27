#                         # check ranking volatility
#                         current_ranking = w_abs.sort()[1] # ranking of different weight groups
#                         logtmp = "    Rank_volatility: "
#                         v = []
#                         cnt_reward = 0
#                         for i in range(n_wg):
#                             chl = current_ranking[i]
#                             self.ranking[name][chl].append(i)
#                             volatility = self._get_volatility(self.ranking[name][chl])
#                             logtmp += "%d " % volatility
#                             v.append(volatility)

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
