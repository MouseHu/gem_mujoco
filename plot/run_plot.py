import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from plot.plot_utils import *

root = "C:/Users/Mouse Hu/Desktop/GEM/data_ddq_final/"
# fix problem may refer to layer normalization?

if __name__ == "__main__":
    tag = "eval_ep_rewmean"
    env = "hopper"

    # mine_config = "ddq_update_200_lnmlp"
    # mine_config = "ddq_reverse_double"
    # mine_config = "ddq_test_fix_problem_max_step_5"
    mine_config = "ddq_max_step_5_punish"
    # mine_config = "ddq_test_update_200"
    # mine_config = "ddq_test_fix_problem"
    # mine_config = "ddq_test_fix_problem"
    td3_config = "td3_baseline_3"
    sac_config = "sac_baseline_1"
    sil_config = "td3sil_baseline_0"
    ddpg_config = "ddpg_baseline_"
    mine_data = data_read(
        paths=[root + 'ddq/run-{}_{}_{}_tb-tag-{}.csv'.format(env, mine_config, i, tag) for i in [0,1,2,3,4]])
        # paths=[root + 'revisited/run-{}_{}_{}_tb-tag-{}.csv'.format(env, mine_config, i, tag) for i in [0,1,2,3,4]])
    td3_data = data_read(
        paths=[root + 'td3/run-{}_{}_{}_tb-tag-{}.csv'.format(env, td3_config, i, tag) for i in range(5)])
    # paths = [root + 'td3/{}_TD3_baseline_{}.csv'.format(env, i) for i in range(10)])

    sil_data = data_read(
        paths=[root + 'td3sil/run-{}_{}_{}_tb-tag-{}.csv'.format(env, sil_config, i, tag) for i in [0,1,2,3,4]])
    ddpg_data = data_read(
        paths=[root + 'ddpg/run-{}_{}_{}_tb-tag-{}.csv'.format(env, ddpg_config, i, tag) for i in range(5)])
    # paths = [root + 'ddpg/{}_DDPG_baseline_{}.csv'.format(env, i) for i in range(10)])
    sac_data = data_read(
        paths=[root + 'sac/run-{}_{}_{}_tb-tag-{}.csv'.format(env, sac_config, i, tag) for i in range(5)])
    # paths=[root + 'sac/{}_SAC_baseline_{}.csv'.format(env, i) for i in range(10)])

    # datas = [mine_data,td3_data]
    datas = [mine_data, td3_data, sil_data, sac_data, ddpg_data]
    legends = ['GEM', 'TD3', "TD3+SIL", "SAC", "DDPG"]
    # datas = [mine_data_new_intr, mine_data_exploit_only]
    # legends = ['MetaCURE', 'MetaCURE Without Exploitation Policy']
    plot_all(datas, legends, 1)
    # plt.title('{}-{}'.format(env,tag), size=20)
    plt.title('Hopper-v2', size=30)
    # plt.plot(mine_data[0], np.ones(mine_data[0].shape) * 4.02, color='olive', linestyle='--', linewidth=2,label='EPI')
    legend()
    plt.show()
