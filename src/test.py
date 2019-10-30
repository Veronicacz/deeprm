import gym
from gym import spaces
# import gym_foo
import numpy as np 
# import universe

env = gym.make('foo-v0')
# env = gym.make('CartPole-v0')

# print(spaces.Discrete(2).0)

print(env.observation_space)  # Box(20, 124)   
print(env.action_space)  # Box(5,)

# a = np.array([[1, 2], [3, 4]])
# print(a.shape)
# b = np.array([[1, 2], [3, 4]]).reshape(2,2)
# print(a)
# print(b)
