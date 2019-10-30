import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent
import action_space


class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions=1e6, k_ratio=1):
        super().__init__(env)
        self.experiment = env.spec.id
        if self.continious_action_space:
            self.action_space = action_space.Space(self.low, self.high, max_actions)
            # print(self.low) #[0 0 0 0 0]
            # print(self.high)#[1 1 1 1 1]
            # print(max_actions) #1000/64
            max_actions = self.action_space.get_number_of_actions()
            #self.max_actions = max_actions
            # print(max_actions) #1024/32
        else:
            # tem = int(env.action_space.n)
            max_actions = int(env.action_space.n) #int((tem+1)/2) #int(env.action_space.n)
            #self.max_actions = max_actions
            self.action_space = action_space.Discrete_space(max_actions)

        self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))

    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def act(self, state):
        # print("actor_state")
        # print(state.shape)   # (2480, )
        # taking a continuous action from the actor
        proto_action = super().act(state)

        print("proto_action")
        # print(proto_action.shape) # (1, 1)
        print(proto_action)  # [[0.0017871]]
        # exit()

#        proto_action = np.argmax(proto_action)


#        print("proto_action")
#        print(proto_action) 
        # if self.k_nearest_neighbors < 1:
        #     return proto_action

        # return the best neighbor of the proto action
        # return self.wolp_action(state, proto_action)
        return proto_action

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)[0]
        # print("test3a")
        # print(actions.shape) # (100,1)  /(102, 5) /(3,1)
        # # print(actions.shape[0])  #100 / 102
        # print(len(actions))  # 100/102   #3
        # print(state.shape)  # (4,)/ (20, 124) / (20,34)

        # deleted by zc 07/05  # data file
        # self.data_fetch.set_ndn_action(actions[0].tolist())


        #modify 07/05
        # make all the state, action pairs for the critic
        # states = np.tile(state, [len(actions), 1, 1])  # ((20, 124), [(102, 1)])  ->>> (2040, 124)  #modify 07/05

        states = np.reshape(state, [1, self.observation_space_size])


        # numpy.tile(A, reps)
        # 
        # Construct an array by repeating A the number of times given by reps.
        # print("test3b")
        # print(states.shape) #   (100, 4) /(2040, 124) /(60, 34)
        # evaluate each pair through the critic

        # tem_states = states.reshape((1, 2480))

        tem_states = np.tile(states, [len(actions), 1])

        # print("tem_states")

        # print(tem_states.shape)  # (1, 680)
        # print(tem_states.ndim)  # 2
        # print("array size")
        # print(self.observation_space_size)#*self.k_nearest_neighbors)
        # exit()

        # tem_states = states.reshape((1, self.observation_space_size))#*self.k_nearest_neighbors))  #modify 07/05
        # actions_evaluation = self.critic_net.evaluate_critic(states, actions)

        actions_evaluation = self.critic_net.evaluate_critic(tem_states, actions)
        # find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation)
        # return the best action
        return actions[max_index]
