# matplotlib.use('TkAgg')
from plot.plot_utils import *

root = "C:/Users/Mouse Hu/Desktop/GEM/data_ddq_final/"
if __name__ == "__main__":
    # tag = "eval_qs_difference"
    tag = "eval_ep_rewmean"
    env = "ant"

    # mine_config = "ddq_test_fix_problem_nodelay"
    # mine_config = "ddq_test_update_200"
    # mine_config = "ddq_max_step_5_punish"
    # mine_config = "ddq_100000_update_200_lnmlp_eval"
    mine_config = "ddq_update_200_lnmlp"
    no_double_config = "ddq_100000_update_200_lnmlp_no_double_eval"
    no_conservative_config = "ddq_100000_update_200_lnmlp_alpha_1_eval"
    # gae_config = "ddq_update_200_lnmlp_beta_0.95_real"
    gae_config = "ddq_update_200_lnmlp_beta_0.95"
    ot_config = "ot_length_3"
    mine_data = data_read(
        paths=[root + 'revisited/run-{}_{}_{}_tb-tag-{}.csv'.format(env, mine_config, i, tag) for i in range(5)])
    # paths=[root + 'rebuttal_overestimation_2/run-{}_{}_{}_tb-tag-{}.csv'.format(env, mine_config, i, tag) for i in range(5)])
    # sil_data = data_read(
    #     paths=[root + 'td3sil/run-{}_{}_{}_tb-tag-{}.csv'.format(env, sil_config, i, tag) for i in range(5)])
    #
    # redq_data = data_read(
    #     paths=[root + 'td3redq/run-{}_{}_{}_tb-tag-{}.csv'.format(env, redq_config, i, tag) for i in range(5)])
    #
    no_double_data = data_read(
        paths=[root + 'rebuttal_overestimation_2/run-{}_{}_{}_tb-tag-{}.csv'.format(env, no_double_config, i, tag) for i
               in range(5)])

    no_converative_data = data_read(
        paths=[root + 'rebuttal_overestimation_2/run-{}_{}_{}_tb-tag-{}.csv'.format(env, no_conservative_config, i, tag)
               for i in range(5)])

    gae_data = data_read(
        paths=[root + 'revisited/run-{}_{}_{}_tb-tag-{}.csv'.format(env, gae_config, i, tag) for i in range(5)])

    ot_data = data_read(
        paths=[root + 'ot/run-{}_{}_{}_tb-tag-{}.csv'.format(env, ot_config, i, tag) for i in range(5)])
    # datas = [mine_data,td3_data]
    # datas = [mine_data, sil_data, redq_data, no_double_data]
    datas = [mine_data, no_double_data, no_converative_data,gae_data,ot_data]
    # legends = ['GEM', 'TD3+SIL', 'REDQ', 'GEM without TBP']
    legends = ['GEM', 'GEM No TBP', 'GEM No Conservative Update', 'GEM - TBP + lambda-return',
               'GEM - TBP + optimality tightning']
    # datas = [mine_data_new_intr, mine_data_exploit_only]
    # legends = ['MetaCURE', 'MetaCURE Without Exploitation Policy']
    # plt.figure(figsize=(12, 8))
    plot_all(datas, legends, 1)
    # plt.title('{}-{}'.format(env,tag), size=20)
    plt.title('Performance:Ant-v2', size=50)
    # plt.plot(mine_data[0], np.ones(mine_data[0].shape) * 4.02, color='olive', linestyle='--', linewidth=2,label='EPI')
    legend()

    plt.savefig("C:/Users/Mouse Hu/Desktop/GEM/{}_overestimation.pdf".format(env), bbox_inches='tight')
    plt.show()
