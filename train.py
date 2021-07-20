import rlschool
from model import QuadModel
from agent import MyAgent
import parl
import paddle
import numpy as np
from parl.utils import logger, summary, ReplayMemory
from action_mapping import my_action_mapping

MAX_STEPS = 1000000
LEARN_FREQ = 10
MEMORY_WARMUP_SIZE = 1e4
BATCH_SIZE = 256
MEMORY_SIZE = 1e6
GAMMA = 0.99
E_GREED = 0.05
TAU = 0.01     #TODO 软更新参数，参数可能太小，收敛速度太慢了
ACTOR_LR = 0.001
CRITIC_LR = 0.001
UPDATE_TARGET = 20
TEST_FREQ = 10000
REWARD_SCALE = 0.01
NOISE = 0.05
RESTORE = False
SAVE_MODEL = True
RESTORE_PATH = 'model_dir/02/steps_1000932.ckpt'
SAVE_PATH = 'model_dir/03/steps_{}.ckpt'

def run_episode(env, agent, rpm):

    action_dim = env.action_space.shape[0]+1
    obs = env.reset()
    total_reward = 0
    n_step = 0
    while True:
        # if total_step >=50000:
        # env.render()
        n_step += 1

        if rpm.size() <MEMORY_WARMUP_SIZE:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)

        # 高斯噪声
        # np.random.normal()  正态分布，第一个参数为中心轴，第二个参数为正态分布标准差，越大越矮胖
        # np.clip()    截取函数，第一个为被截取的矩阵，后两个为下限和上线，所有超界限的数都会被约束在界限内。
        action = np.random.normal(action, 1)     # TODO 是不是因为噪声太大了模型不收敛
        # print("act:{}".format(action))

        action = np.squeeze(action)


        # print("action:{}".format(action))
        # print("step:{}  action:{}".format(n_step,action))

        temp = np.zeros((1,4))
        temp_1 = np.array([[1.0, 0.0, 0.0, 0.0]])
        temp_2 = np.array([[0.0, 1.0, 0.0, 0.0]])
        temp_3 = np.array([[0.0, 0.0, 1.0, 0.0]])
        temp_4 = np.array([[0.0, 0.0, 0.0, 1.0]])

        temp += list(action)[0]
        temp_1 *= list(action)[1]
        temp_2 *= list(action)[2]
        temp_3 *= list(action)[3]
        temp_4 *= list(action)[4]

        action_f = temp + 0.1 * (temp_1 + temp_2 + temp_3 + temp_4)

        action_f = np.squeeze(action_f)
        action_f = np.squeeze(action_f)
        action_f = np.clip(action_f, -1.0, 1.0)

        action_f = my_action_mapping(action_f, env.action_space.low[0], env.action_space.high[0])

        next_obs, reward, reset, _ = env.step(action_f)
        # print("next_obs:{}   reward:{}    reset:{}".format(next_obs,reward,reset))
        # print("000")
        # print(next_obs)
        # print("111")
        rpm.append(obs, action, reward * REWARD_SCALE, next_obs, reset)
        # print(len(rpm))
        # train
        if(len(rpm)>MEMORY_WARMUP_SIZE) and (n_step % LEARN_FREQ == 0):
            # print("--------------------")
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_reset) = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward,
                        batch_next_obs, batch_reset)

            # print("i have learned it ")



        total_reward += reward
        obs = next_obs
        if  reset:
            break

    return total_reward, n_step

def run_evaluate_episodes(env, agent , eval_episode=3, render = False):
    eval_reward = []
    for _ in range(eval_episode):
        obs = env.reset()
        episode_reward = 0
        while True:
            if RESTORE:
                env.render()


            action = agent.predict(obs)
            action = np.squeeze(action)

            temp = np.zeros((1, 4))
            temp_1 = np.array([[1.0, 0.0, 0.0, 0.0]])
            temp_2 = np.array([[0.0, 1.0, 0.0, 0.0]])
            temp_3 = np.array([[0.0, 0.0, 1.0, 0.0]])
            temp_4 = np.array([[0.0, 0.0, 0.0, 1.0]])

            temp += list(action)[0]
            temp_1 *= list(action)[1]
            temp_2 *= list(action)[2]
            temp_3 *= list(action)[3]
            temp_4 *= list(action)[4]

            action_f = temp + 0.1 * (temp_1 + temp_2 + temp_3 + temp_4)
            action_f = np.squeeze(action_f)
            action_f = np.squeeze(action_f)
            action_f = np.clip(action_f, -1.0, 1.0)

            action_f = my_action_mapping(action_f, env.action_space.low[0], env.action_space.high[0])

            obs, reward, reset, _ =env.step(action_f)
            episode_reward += reward

            if reset:
                break

        eval_reward.append(episode_reward)
        # logger.info("eval_reward: {}".format(episode_reward))
    return np.mean(eval_reward)






def main():
    max_step = MAX_STEPS
    env = rlschool.make_env("Quadrotor", task="hovering_control")
    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]+1

    env.reset()
    # print(dir(env))

    model = QuadModel(obs_dim, action_dim)
    alg = parl.algorithms.DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = MyAgent(alg,E_GREED,action_dim,UPDATE_TARGET)
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, action_dim)

    num_episode = 0
    total_step = 0
    test_flag = 0
    print("*******************************************************")
    if RESTORE:
        cpkt = RESTORE_PATH
        agent.restore(cpkt)
        evaluate_reward = run_evaluate_episodes(env, agent)
        logger.info('Evaluate reward: {}'.format(evaluate_reward))

    else:
        while total_step < max_step:
            num_episode += 1
            reward, steps = run_episode(env, agent, rpm)
            total_step += steps



            if total_step // TEST_FREQ >= test_flag:
                test_flag += 1

                evaluate_reward = run_evaluate_episodes(env, agent)
                logger.info('Episode: {}   step: {}   Test reward: {}'.format(
                    num_episode, total_step, evaluate_reward))
                if SAVE_MODEL:
                    ckpt = SAVE_PATH.format(total_step)
                    agent.save(ckpt)







    # while episode < max_episode:
    #     reward = 0
    #     for i in range(20):
    #         reward += run_episode(agent, env, rpm)
    #         episode += 1
    #         print("*****"+str(episode)+"******")
    #
    #
    #     eval_reward = run_evaluate_episodes(agent,env)
    #     print('episode:{}  Test reward:{}'.format(
    #         episode, eval_reward))

if __name__ == '__main__':
    main()
