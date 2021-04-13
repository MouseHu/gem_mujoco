import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from plot.plot_utils import *

root = "C:/Users/Mouse Hu/Desktop/GEM/data_ddq_final/"
if __name__ == "__main__":
    tag = "eval_ep_rewmean"
    env = "halfcheetah"

    # mine_config = "ddq_test_fix_problem_nodelay"
    # mine_config = "ddq_test_update_200"
    # mine_config = "ddq_max_step_5_punish"
    mine_config = "ddq_update_200_lnmlp"
    gae_config = "ddq_update_200_lnmlp_beta_0.95"

    mine_data = data_read(
        paths=[root + 'revisited/run-{}_{}_{}_tb-tag-{}.csv'.format(env, mine_config, i, tag) for i in range(5)])
    # sil_data = data_read(
    #     paths=[root + 'td3sil/run-{}_{}_{}_tb-tag-{}.csv'.format(env, sil_config, i, tag) for i in range(5)])
    #
    # redq_data = data_read(
    #     paths=[root + 'td3redq/run-{}_{}_{}_tb-tag-{}.csv'.format(env, redq_config, i, tag) for i in range(5)])
    #
    # no_double_data = data_read(
    #     paths=[root + 'amc/run-{}_{}_{}_tb-tag-{}.csv'.format(env, nodouble_config, i, tag) for i in range(5)])

    gae_data = data_read(
        paths=[root + 'revisited/run-{}_{}_{}_tb-tag-{}.csv'.format(env, gae_config, i, tag) for i in range(5)])

    # datas = [mine_data,td3_data]
    # datas = [mine_data, sil_data, redq_data, no_double_data]
    datas = [mine_data, gae_data]
    # legends = ['GEM', 'TD3+SIL', 'REDQ', 'GEM without TBP']
    legends = ['GEM', 'GAE']
    # datas = [mine_data_new_intr, mine_data_exploit_only]
    # legends = ['MetaCURE', 'MetaCURE Without Exploitation Policy']
    plot_all(datas, legends, 1)
    # plt.title('{}-{}'.format(env,tag), size=20)
    plt.title('HalfCheetah-v2', size=30)
    # plt.plot(mine_data[0], np.ones(mine_data[0].shape) * 4.02, color='olive', linestyle='--', linewidth=2,label='EPI')
    legend()
    plt.show()
