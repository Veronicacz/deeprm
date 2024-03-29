import random
import numpy as np

from util import *

import gym
from gym.spaces import Box, Discrete

from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from tensorflow_grad_inverter import grad_inverter

from collections import deque

"""Some agent implementations including stevenpjg's DDPG agent"""


class Agent:

    def __init__(self, env):



        # checking state space
        if isinstance(env.observation_space, Box):
            self.observation_space_size = env.observation_space.shape[0]
            self.observation_space_size1 = env.observation_space.shape[0] #env.observation_space.shape[0] #20
            self.observation_space_size2 = env.observation_space.shape[1] #124?
            self.observation_space_size = self.observation_space_size1*self.observation_space_size2 # 2480
            # self.observation_space_size = self.observation_space_size2
            # print("test1")
            # print(self.observation_space_size) #4
        else:
            self.observation_space_size = env.observation_space.n

        # checking action space
        if isinstance(env.action_space, Box):
            self.action_space_size = env.action_space.shape[0]
            # print("test2")
            # print(self.action_space_size)  #1/5
            self.continious_action_space = True
            self.low = env.action_space.low
            self.high = env.action_space.high
        else:  # dont need in this implementation
            tem = int(env.action_space.n)
            self.action_space_size = tem #int((tem+1)/2) #env.action_space.n #1
            self.action_space_size_ac = tem #int((tem+1)/2) #env.action_space.n
            self.continious_action_space = False
            #tem_a = np.zeros(self.action_space_size)
            #tem_a[0] = 1
            #self.low = tem_a #
            self.low = np.zeros(self.action_space_size) #np.array([0])
            self.high = np.ones(self.action_space_size)#np.array([env.action_space.n])
            # print("test2")
            # print(self.action_space_size)
            # print(self.high) # [4]
            # exit()

    def act(self, state):
        pass

    def observe(self, episode):
        pass

    def get_name(self):
        return 'Agent'

    def _np_shaping(self, array, is_state):

        # print("test4")
        # # print(array)  #[ 0.00592778 -0.05883199 -0.5027318   0.71983569]
        # print(array.shape) # (20, 124) /(1,)/(4,)  # (2480, )
        # print(array.shape[1])

        # number_of_elements = array.shape[0] if len(array.shape) > 1 else 1
        # size_of_element = self.observation_space_size if is_state else self.action_space_size

        number_of_elements = array.shape[0] if len(array.shape) > 2 else 1
        size_of_element = self.observation_space_size if is_state else self.action_space_size_ac
        # size_of_element = self.observation_space_size2 if is_state else self.action_space_size
        # print(number_of_elements)  # 1/1/20     # 1
        # print(size_of_element)  # 1/4/2480     # 2480

        

        res = np.array(array)
        res.shape = (number_of_elements, size_of_element)  # (1, 4)/(1, 2480)
        # print(res) # [[ 0.00592778 -0.05883199 -0.5027318   0.71983569]]
        return res


class DDPGAgent(Agent):
    ''' stevenpjg's implementation of DDPG algorithm '''

    REPLAY_MEMORY_SIZE =  10000 #1000 #10000  # 1000 # 20 #
    BATCH_SIZE = 1000 #100 #1000 # 64  #10  # 
    GAMMA = 0.99


    def __init__(self, env, is_batch_norm=False, is_grad_inverter=True):
        super().__init__(env)

        if is_batch_norm:
            self.critic_net = CriticNet_bn(self.observation_space_size,
                                           self.action_space_size_ac)
            self.actor_net = ActorNet_bn(self.observation_space_size,
                                         self.action_space_size_ac)

            #self.critic_net = CriticNet_bn(self.observation_space_size2,
             #                              self.action_space_size)
            #self.actor_net = ActorNet_bn(self.observation_space_size2,
             #                            self.action_space_size)

        else:
            self.critic_net = CriticNet(self.observation_space_size,
                                        self.action_space_size_ac)
            self.actor_net = ActorNet(self.observation_space_size,
                                       self.action_space_size_ac)
            #self.critic_net = CriticNet(self.observation_space_size2,
             #                           self.action_space_size)
            #self.actor_net = ActorNet(self.observation_space_size2,
             #                         self.action_space_size)

        self.is_grad_inverter = is_grad_inverter
        self.replay_memory = deque()

        self.time_step = 0

        action_max = np.array(self.high).tolist()
        action_min = np.array(self.low).tolist()
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)

    def add_data_fetch(self, df):
        self.data_fetch = df

    def get_name(self):
        return 'DDPG' + super().get_name()

    def act(self, state):
        
        state = self._np_shaping(state, True)
        # print("reshape:")
        # print(state.shape)  # (1,4)/(1, 2480)
        # exit()
        result = self.actor_net.evaluate_actor(state).astype(float)
        # print(result.shape)

        # deleted by zc 07/05  # data file
        # self.data_fetch.set_actors_action(result[0].tolist())

        return result

    def observe(self, episode):
        episode['obs'] = self._np_shaping(episode['obs'], True)
        episode['action'] = self._np_shaping(episode['action'], False)
        episode['obs2'] = self._np_shaping(episode['obs2'], True)
        self.add_experience(episode)

    def add_experience(self, episode):
        self.replay_memory.append(episode)

        self.time_step += 1
        if len(self.replay_memory) > type(self).REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > type(self).BATCH_SIZE:
            res = self.train()
            return res
        else:
            return None

    def minibatches(self):
        batch = random.sample(self.replay_memory, type(self).BATCH_SIZE)
        # state t
        state = self._np_shaping(np.array([item['obs'] for item in batch]), True)
        # action
        action = self._np_shaping(np.array([item['action'] for item in batch]), False)
        # reward
        reward = np.array([item['reward'] for item in batch])
        # state t+1
        state_2 = self._np_shaping(np.array([item['obs2'] for item in batch]), True)
        # doneA
        done = np.array([item['done'] for item in batch])
        # doneB
#        done1 = np.array([item['done1'] for item in batch])

        return state, action, reward, state_2, done

    def train(self):
        # sample a random minibatch of N transitions from R
        state, action, reward, state_2, done = self.minibatches()

        actual_batch_size = len(state)

        target_action = self.actor_net.evaluate_target_actor(state_2)

        # Q'(s_i+1,a_i+1)
        q_t = self.critic_net.evaluate_target_critic(state_2, target_action)

        y = []
        for i in range(0, actual_batch_size):

#            print("done test:")
#            print(done[i])

            if done[i]:
                y.append(reward[i])
            else:
                y.append(reward[i] + type(self).GAMMA * q_t[i][0])  # q_t+1 instead of q_t

        y = np.reshape(np.array(y), [len(y), 1])

        # Update critic by minimizing the loss
        self.critic_net.train_critic(state, action, y)

        # Update actor proportional to the gradients:
        # action_for_delQ = self.act(state)  # was self.evaluate_actor instead of self.act
        action_for_delQ = self.actor_net.evaluate_actor(state)  # dont need wolp action

        if self.is_grad_inverter:
            del_Q_a = self.critic_net.compute_delQ_a(state, action_for_delQ)  # /BATCH_SIZE
            del_Q_a = self.grad_inv.invert(del_Q_a, action_for_delQ)
        else:
            del_Q_a = self.critic_net.compute_delQ_a(state, action_for_delQ)[0]  # /BATCH_SIZE

        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(state, del_Q_a)

        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
