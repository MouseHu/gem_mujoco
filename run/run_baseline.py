import time
import os
import tensorflow as tf
import json
from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds, boolean_flag
# from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
# from stable_baselines.sac.policies import LnMlpPolicy as SACLnMlpPolicy

from stable_baselines.ddpg import DDPG
from stable_baselines.sac import SAC
from stable_baselines.td3 import TD3
from stable_baselines.td3.td3_sil import TD3SIL
from stable_baselines.td3.td3_redq import TD3REDQ
from run.run_util import parse_args, create_action_noise, create_env, save_args


def run(env_id, seed, layer_norm, evaluation, agent, delay_step, gamma=0.99, **kwargs):
    # Create envs.
    env = create_env(env_id, delay_step, str(0))
    print(env.observation_space, env.action_space)
    if evaluation:
        eval_env = create_env(env_id, delay_step, "eval_env")
    else:
        eval_env = None

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))

    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed + 1)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()

    policy = 'LnMlpPolicy' if layer_norm else 'MlpPolicy'
    if agent == "DDPG":
        model = DDPG(policy=policy, env=env, eval_env=eval_env, gamma=gamma, nb_eval_steps=5, batch_size=64,
                     nb_train_steps=100, nb_rollout_steps=100,
                     actor_lr=1e-4, critic_lr=1e-3, critic_l2_reg=1e-2,
                     tau=0.001, normalize_observations=True,
                     action_noise=create_action_noise(env, "ou_0.2"), buffer_size=int(1e6),
                     verbose=2, n_cpu_tf_sess=10,
                     policy_kwargs={"layer": [400, 300]})
    elif agent == "TD3":
        model = TD3(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=100,
                    tau=0.005, policy_delay=2,
                    action_noise=create_action_noise(env, "normal_0.1"), buffer_size=50000, verbose=2, n_cpu_tf_sess=10,
                    policy_kwargs={"layer": [400, 300]})
    elif agent == "TD3SIL":
        model = TD3SIL(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=100,
                       tau=0.005, policy_delay=2,
                       action_noise=create_action_noise(env, "normal_0.1"), buffer_size=50000, verbose=2,
                       n_cpu_tf_sess=10,
                       policy_kwargs={"layer": [400, 300]})
    elif agent == "TD3REDQ":
        model = TD3REDQ(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=100,
                        tau=0.005, policy_delay=2,
                        action_noise=create_action_noise(env, "normal_0.1"), buffer_size=50000, verbose=2,
                        n_cpu_tf_sess=10,
                        policy_kwargs={"layer": [400, 300]})
    elif agent == "SAC":
        model = SAC(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=64,
                    action_noise=create_action_noise(env, "normal_0.1"), buffer_size=int(1e6), verbose=2,
                    n_cpu_tf_sess=10,
                    policy_kwargs={"layer": [256, 256]})
    else:
        raise NotImplementedError

    print("model building finished")
    model.learn(total_timesteps=kwargs['num_timesteps'])

    env.close()
    if eval_env is not None:
        eval_env.close()

    logger.info('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_LOGDIR"] = os.path.join(os.getenv("OPENAI_LOGDIR"), args["comment"])
    save_args(args)
    logger.configure()
    # Run actual script.
    run(**args)
