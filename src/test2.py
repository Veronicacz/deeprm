# for i in range(2):
#     for j in range(3, 5):
#         if j <= 3:
#             # continue
#             break
#         print("j = " + str(j)) 
#     print("i = " + str(i))


# import numpy as np

# tem_a = [1, 2, 3, 4]
# print tem_a[:3]
# print np.any(tem_a[:]>0)
# a = np.array(tem_a)
# # print a 

# size = 4
# ad = np.zeros((2, 2))
# print len(ad)

# print a[0:2]

# print a[1]
# index1 = np.where(a==1)[0]
# print index1
# print a[index1]
# # index1 = a.index(1)

# index2 = a.tolist().index(2)
# print index2

# b = a-1
# # print b

# # print min(a)

# # np.isclose(a, b)
# tem_b = b.tolist()

# print a.size

# x = np.zeros((3, 5, 2), dtype=np.complex128)

# print x.size
# print x.shape[0]  # 3

# # # Python 3

# import gym

# # env = gym.make('InvertedPendulum-v2')

# # env = gym.make('Ant-v2')

# env = gym.make('foo-v0')

# # print(env.observation_space)  # Box(111,)
# # print(env.action_space)  # Box(8,)

# # print(env.spec.timestep_limit)  # 1000

# # print(env.action_space.low)  #[-1. -1. -1. -1. -1. -1. -1. -1.]
# # print(type(env.action_space.low))  # <class 'numpy.ndarray'>

# # print(env.action_space.high)  # [1. 1. 1. 1. 1. 1. 1. 1.]
# # print(type(env.action_space.low))   # <class 'numpy.ndarray'>

# # print(type(env.observation_space))  # <class 'gym.spaces.box.Box'>

# # print(env.observation_space.shape[0])  # 4
# # # print(env.observation_space.n)

# # print(env.action_space.shape[0])  # (1,)  # 1

# # print(env.observation_space.shape) # (4, )

# # # Box(4,)
# # # Box(1,)
# # # [-3.]
# # # [3.]

# print(env.action_space.low)  # [-3.] / [0]
# print(env.action_space.high) # [3.] / [4]
# print(type(env.action_space.low))  # <class 'numpy.ndarray'> / <class 'numpy.ndarray'>


# print(1e6*0.1)


import numpy as np
import math

num_clu = 3
subtask_order = np.zeros(num_clu)
tem_clu_arrange = np.arange(1, num_clu+1)
np.random.shuffle(tem_clu_arrange)
for i in range(num_clu):
	subtask_order[i] = tem_clu_arrange[i]

print(subtask_order)

des = np.argmax(subtask_order)
print(des)

print(subtask_order[0:2])
ind = np.where(subtask_order == 3)



print(ind)

print(subtask_order[ind])

print(tem_clu_arrange[ind])

print(math.ceil(2))
# # import gym


# # env = gym.make('rs-v0')
# # # env = gym.make('foo-v0')

# a = np.array([0, 1, 2])
# b = np.tile(a, (2, 1))
# print(b)
# print(b.shape)

# a = np.array([[0,1,2], [1,2,3]])
# print(a.ndim)
# b = np.reshape(a, [1, 6])
# print(b)
# c = np.tile(b, (2, 1))
# print(c)
# print(c.shape)


# for i in range(4):
# 	if i > 1:
# 		print(i)
# 		break


# tem_b = []
# tem_b.append(False)

# tem_b.append(False)

# print(tem_b)


# tem_b[0] = True
# print(tem_b)

# while not tem_b[1]:
# 	print("test")
# 	tem_b[1] = True


a = np.random.randint(1, 2)
print(a)

b = np.zeros((5, 5))
print(b)
print(len(b))
print(len(b[0]))


tem_a = np.zeros((4, 4), dtype = int)
for i in range(len(tem_a)):
	for j in range(i, len(tem_a[i])):
		if j != i:
			tem_r = np.random.randint(1, 5+1)
			tem_a[i][j] = tem_r
			tem_a[j][i] = tem_r
		else:
			tem_a[i][j] = 0

print(tem_a)

for i in range(6):
	if i>3:
		print(i)
		break


from collections import deque

queue = deque()
# queue = [None] * 5
queue.append(None)
queue.append(None)

queue[0] = 5
print(queue)
queue[1] = 4
b = queue.popleft()
print(queue)
print(b)

q = [None] * 4
# for i in range(4):
# 	q[i] = deque()
# 	for j in range(5):
# 		q[i].append(None)


print(q)

z = np.ones((2, 3)) * 4
print(z)

a = np.ones(5)
b = a
b = a - 2
print(b)
print(a)


c = np.zeros((4,4,2,1))
print(c)

print(c[1][2])
print(c[1,2,:,:])

# self.trans_canvas = np.zeros((self.num_clu, self.num_clu,self.time_horizon, pa.max_transmiss_rate))
        # self.canvas = np.zeros((self.num_clu, self.time_horizon, self.res_slot))


for i in range(5):
	for j in range(4):
		if j > 2:
			print("j > 2")
			break


# for i in range(5):
# 	for j in range(4):
# 		if j > 2:
# 			print("j > 2")
# 			break


c = np.array([[[1,1,2], [1,2,3]], [[1,1,2], [1,2,3]]])

if np.all(c[:] > 0):
	print("true")


for t in range(0, -2):
	print(t)


a = np.zeros(5)
b = np.ones(5)
#if a > b:
#	print("True")
#elif a < b:
#	print("False")
a = []
a.append(1)
a.append(2)
a.append(3)
a.append(4)

inde = a.index(max(a))
print(inde)
print(a[inde])

b = np.array(a)
print(b[0:3])

for i in range(0, 5):
	if i > 0:
		print(i)
		if i > 2:
			# print(i)
			break


t = []
t.append(1)
t.append(True)
print(t)

a = np.zeros(6)
a[0] = 1
bc = a
print(bc)