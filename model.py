import parl
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from parl.core.fluid import layers

class QuadModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(QuadModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model.forward(obs)

    def value(self, obs, action):
        return self.critic_model.forward(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        # self.l1 = nn.Linear(obs_dim, 400)
        # self.l2 = nn.Linear(400, 300)
        # self.l3 = nn.Linear(300, action_dim)

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, obs):
        # a_1 = F.relu(self.l1(obs))
        # a_2 = F.relu(self.l2(a_1))
        # return paddle.tanh(self.l3(a_2))

        a_1 = F.relu(self.l1(obs))
        a_2 = F.relu(self.l2(a_1))
        return paddle.tanh(self.l3(a_2))



class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400+ action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, obs, action):
        # concat = layers.concat([obs, action], axis=1)
        # hid_1 = F.relu(self.l1(concat))
        # hid_2 = F.relu(self.l2(hid_1))
        # Q = self.l3(hid_2)
        # return Q

        x = F.relu(self.l1(obs))
        concat = layers.concat([x, action], axis=1)
        hid_1 = F.tanh(self.l2(concat))
        Q = self.l3(hid_1)
        return Q

