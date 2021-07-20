import parl
import paddle
import paddle.nn as nn
import numpy as np

class MyAgent(parl.Agent):
    def __init__(self,algorithm,e_greed,act_dim,update_target_step):
        super(MyAgent, self).__init__(algorithm)
        self.e_greed = e_greed
        self.act_dim = act_dim
        self.update_target_step = update_target_step
        self.global_episode = 0
        self.num_update_target = 0
        # 没有加入e-greedy—decrement
    def sample(self, obs):

        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, 0.1, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

        # obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        # action, _ = self.alg.sample(obs)
        # action_numpy = action.cpu().numpy()[0]
        # sample = np.random.random()
        # if sample < self.e_greed:
        #     act = np.random.sample(self.act_dim,)*1.99 - 0.995     #TODO IS OK?
        #     # print("act1")
        # else:
        #     act = self.predict(obs)
        #     # print("act3")
        # return action_numpy

    def predict(self,obs):
        obs = paddle.to_tensor(obs.reshape(1,-1),dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy


    def learn(self, obs, action, reward, next_obs, is_over):
        # print("**************************")
        # print("*******************************")

        is_over = np.expand_dims(is_over, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        is_over = paddle.to_tensor(is_over, dtype='float32')
        # print(obs)
        # print(action)

        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs, is_over)

        self.global_episode += 1
        if self.global_episode % self.update_target_step == 0:
            self.alg.sync_target()
            self.num_update_target += 1
            #print("the {}th update  target".format(self.num_update_target))

        return critic_loss, actor_loss