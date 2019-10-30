#!/usr/bin/python3
import gym

import numpy as np
import sys

from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer
import shutil
import time

import random 

from env import rs

# from ou import OU

# ou = OU()


def run(episodes=100, #5, # 2500,
        render=False,
        # experiment='InvertedPendulum-v2',
        experiment='rs-v0',
        # 'foooc-v0',
        max_actions=6,   #1000,
        runtime = 1000,
        knn=1):  # 0.1  # 1

    env = gym.make(experiment)

    # print(env.observation_space)
    print(env.observation_space.shape[0])  # 20
    print(env.observation_space.shape)  # (20, 124)
    # print(env.observation_space.shape[1])
    print(env.action_space.shape) #()
    print(env.action_space.n) # 6
    # print(env.action_space)
    # print(env.observation_space.n)
    # print(env.action_space.n)


    max_steps = env.spec.max_episode_steps #env.spec.timestep_limit

    print("max steps:")
    print(max_steps)  # 1000  ---> 300   ---> 500   ->>>> 100
#    print("end")

    steps = 200 #500 #1000 # 15 # 100

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)

    timer = Timer()

    # data = util.data.Data()
    # data.set_agent(agent.get_name(), int(agent.action_space.get_number_of_actions()),
    #                agent.k_nearest_neighbors, 3)
    # data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)

    # agent.add_data_fetch(data)
    # print(data.get_file_name())

    full_epoch_timer = Timer()
    reward_sum = 0

    reward_sum_s = 0

    # random.seed(2)
    ransta1 = np.random.RandomState(123)
    epsilon = 1

    env_s = rs.RsEnv()

#     for ep in range(100): #(episodes):

#         timer.reset()
#         # observation_tem = env.reset()
#         observation_s = env_s.reset()

# #        env.jobsave()

# #        tem_f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len.txt"
#  #       tem_f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size.txt"
# #
# #        f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len" + str(ep) + ".txt"
# #        f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size" + str(ep) + ".txt"

# #        shutil.copy(tem_f1, f_1)
# #        shutil.copy(tem_f2, f_2)

#         total_reward_s = 0
#         count_s = 0
#         print('SJF: Episode ', ep, '/', episodes - 1, 'started...', end='')
#         for t in range(steps):

#             #if render:
#              #   env.render()

#             #if t == 1: env.render()


#             # print("state")
#             # print(observation_tem.shape)  #(20, 124)
#             # print(observation_tem.shape[0]) # 4/20
#             # print(observation.shape[1]) #124

#             # tem_len = observation_tem.shape[0]
#             # tem_wid = observation_tem.shape[1]
#             # tem_siz = tem_len*tem_wid

#             # observation = np.reshape(observation_tem, (tem_siz, ))

#             # print("observe")
#             # print(observation.shape) # (2480, )

#             # exit()
#             print('SJF: Episode', ep, 'step', t, '/', steps, 'started...')

#             tem_done_s = False


#             while not tem_done_s:

#                # done = []
#                 count_s += 1

#                 # action_s = env_s.get_sjf_workload_action() #env_s.get_sjf_len_action() #env_s.get_sjf_workload_action() #agent.act(observation)
#                 # action_s = env_s.get_sjf_len_action()
#                 action_s = env_s.get_random_action()

#                 # deleted by zc 07/05  # data file
#                 # data.set_action(action.tolist())

#                 # data.set_state(observation.tolist())

#                 prev_observation_s = observation_s

#                 print("action")
#                 # print(action.shape)
#                 print(action_s)

#                 print("action_end")

#                 ### 07/16#####
#     #            noise = max(epsilon, 0) * ouf(action[0], 0.0, 0.5, 0.5,ransta1)

#     #            tem_action = int(action[0] + noise)
#                 # observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)
#     #            if epsilon> 0: epsilon -= 1.0/100
                
#                 tem_action_s = int(action_s[0])
#                 observation_s, reward_s, done_s, info_s = env_s.step(tem_action_s if len(action_s) == 1 else action_s)

#             # deleted by zc 07/05  # data file
#             # data.set_reward(reward)

#                 tem = int(env_s.pa.num_nw+1) #int(env_s.pa.num_nw/2+1)
#                 conc_act = np.zeros(tem) #(env_s.pa.num_nw+1)
#                 for i in range(tem+1): #(env_s.pa.num_nw+1):
#                     if i == tem_action_s:
#                         conc_act[i] = 1


#                 print("done test")
#                 print(reward_s)
#                 print(done_s)
#                 print(info_s)



#                 episode_s = {'obs': prev_observation_s,
#                             'action': conc_act, #action_s,
#                             'reward': reward_s,
#                             'obs2': observation_s,
#                             'done': done_s,
#                             #   'done1': done[1],
#                             't': t}

#                 # agent.observe(episode_s)

#                 # total_reward += reward


#                 # f=open("/Users/Veronica/Dropbox/Cao_Zhi/Paper3/results/0726/sjf_len/action_sjf_len.txt", "a")
#                 # f.write("%s\n" % tem_action_s)
#                 # f.close() 

#                 # job_num = env_s.get_job_num()

#                 # f=open("/Users/Veronica/Dropbox/Cao_Zhi/Paper3/results/0726/sjf_len/job_num.txt", "a")
#                 # f.write("%s\n" % job_num)
#                 # f.close()


#                 # running_num = env_s.running_num()

#                 # bl_size = env_s.get_backlog_num()
#                 # job_num_bl = []
#                 # job_num_bl.append(tem_action_s)
#                 # job_num_bl.append(job_num)
#                 # job_num_bl.append(bl_size)
#                 # job_num_bl.append(running_num)

#                 # work_list = []
#                 # for i in range(env_s.pa.num_nw):
#                 #     job = env_s.job_slot.slot[i]
#                 #     if job != None:
#                 #         tem_load = env_s.get_job_workload(job)
#                 #         work_list.append(tem_load)


#                 # len_list = []
#                 # for i in range(env_s.pa.num_nw):
#                 #     job = env_s.job_slot.slot[i]
#                 #     if job != None:
#                 #         tem_load = env_s.get_job_len(job)
#                 #         len_list.append(tem_load)

#                 # f=open("/Users/Veronica/Dropbox/Cao_Zhi/Paper3/results/0726/sjf_len/job_num_bl_wl.txt", "a")
#                 # f.write("%s\n" % job_num_bl)
#                 # f.write("%s\n" % work_list)
#                 # f.write("%s\n" % len_list)
#                 # f.close()                

#                 tem_done_s = info_s


#                 if tem_done_s:
#                     total_reward_s += reward_s


#                 if done_s:
#                     print("done")
#                     print(done_s)

#             if ep == episodes -1:
#                 f=open("/home/veronica/Desktop/env/job_profile/0726/sjf/lastepreward.txt", "a")
#                 f.write("%s\n" % reward_s)
#                 f.close() 

#             if done_s or (t == steps - 1):
#                 t += 1
#                 reward_sum_s += total_reward_s
#                 # time_passed = timer.get_time()
#                 print("done:", done_s)
#                 # print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
#                 #                                                             time_passed, round(
#                 #                                                                 time_passed / t),
#                 #                                                             round(reward_sum / (ep + 1))))


#                 f2=open("/home/veronica/Desktop/env/job_profile/0726/sjf/eachreward.txt", "a")
#                 f2.write("%s\n" % total_reward_s)
#                 f2.close()    


#                 f1=open("/home/veronica/Desktop/env/job_profile/0726/sjf/totalreward.txt", "a")
#                 f1.write("%s\n" % reward_sum_s)
#                 f1.close()                    

#                 # deleted by zc 07/05  # data file
#                 # data.finish_and_store_episode()


#             #    if ep == 1 and done: exit()


#                 break




# # # gym train network

#     exit()

#     for ep in range(episodes):

#         timer.reset()
#         # observation_tem = env.reset()
#         observation = env.reset()

# #        env.jobsave()

# #        tem_f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len.txt"
#  #       tem_f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size.txt"
# #
# #        f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len" + str(ep) + ".txt"
# #        f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size" + str(ep) + ".txt"

# #        shutil.copy(tem_f1, f_1)
# #        shutil.copy(tem_f2, f_2)

#         total_reward = 0
#         count = 0
#         print('RL: Episode ', ep, '/', episodes - 1, 'started...', end='')
#         for t in range(steps):

#             #if render:
#              #   env.render()

#             #if t == 1: env.render()


#             # print("state")
#             # print(observation_tem.shape)  #(20, 124)
#             # print(observation_tem.shape[0]) # 4/20
#             # print(observation.shape[1]) #124

#             # tem_len = observation_tem.shape[0]
#             # tem_wid = observation_tem.shape[1]
#             # tem_siz = tem_len*tem_wid

#             # observation = np.reshape(observation_tem, (tem_siz, ))

#             # print("observe")
#             # print(observation.shape) # (2480, )

#             # exit()
#             print('RL: Episode', ep, 'step', t, '/', steps, 'started...')

#             tem_done = False


#             while not tem_done:

#                # done = []
#                 count += 1

#                 action = agent.act(observation)


#                 # deleted by zc 07/05  # data file
#                 # data.set_action(action.tolist())

#                 # data.set_state(observation.tolist())

#                 prev_observation = observation
#                 print("action")
#                 # print(action.shape)
#                 print(action)
                

#                 ### 07/16#####
#     #            noise = max(epsilon, 0) * ouf(action[0], 0.0, 0.5, 0.5,ransta1)

#     #            tem_action = int(action[0] + noise)
#                 # observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)
#     #            if epsilon> 0: epsilon -= 1.0/100
                
#                 tem_action = np.argmax(action) #int(action[0])
#                 print(tem_action)
#                 print("action_end")
#                 observation, reward, done, info = env.step(tem_action) # if len(action) == 1 else action)

#             # deleted by zc 07/05  # data file
#             # data.set_reward(reward)


#                 print("done test")
#                 print(reward)
#                 print(done)
#                 print(info)



#                 episode = {'obs': prev_observation,
#                             'action': action,
#                             'reward': reward,
#                             'obs2': observation,
#                             'done': done,
#                             #   'done1': done[1],
#                             't': t}

#                 agent.observe(episode)

#                 # total_reward += reward

#                 tem_done = info


#                 if tem_done:
#                     total_reward += reward


#                 if done:
#                     print("done")
#                     print(done)

#             if ep == episodes -1:
#                 f=open("/home/veronica/Desktop/env/job_profile/0726/normal/lastepreward.txt", "a")
#                 f.write("%s\n" % reward)
#                 f.close() 

#             if done or (t == steps - 1):
#                 t += 1
#                 reward_sum += total_reward
#                 time_passed = timer.get_time()
#                 print("done:", done)
#                 print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
#                                                                             time_passed, round(
#                                                                                 time_passed / t),
#                                                                             round(reward_sum / (ep + 1))))


#                 f2=open("/home/veronica/Desktop/env/job_profile/0726/normal/eachreward.txt", "a")
#                 f2.write("%s\n" % total_reward)
#                 f2.close()    

#                 tem_t_info = []
#                 tem_t_info.append(t)
#                 tem_t_info.append(done)

#                 f=open("/home/veronica/Desktop/env/job_profile/0726/normal/done_info.txt", "a")
#                 f.write("%s\n" % tem_t_info)
#                 f.close()  


#                 f1=open("/home/veronica/Desktop/env/job_profile/0726/normal/totalreward.txt", "a")
#                 f1.write("%s\n" % reward_sum)
#                 f1.close() 

#                 agent.actor_net.save_model(ep)
#                 agent.critic_net.save_model(ep)                   

#                 # deleted by zc 07/05  # data file
#                 # data.finish_and_store_episode()


#             #    if ep == 1 and done: exit()


#                 break

# #     for ep in range(episodes):

# #         timer.reset()
# #         # observation_tem = env.reset()
# #         observation_s = env_s.reset()

# # #        env.jobsave()

# # #        tem_f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len.txt"
# #  #       tem_f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size.txt"
# # #
# # #        f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len" + str(ep) + ".txt"
# # #        f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size" + str(ep) + ".txt"

# # #        shutil.copy(tem_f1, f_1)
# # #        shutil.copy(tem_f2, f_2)

# #         total_reward_s = 0
# #         count_s = 0
# #         print('SJF2: Episode ', ep, '/', episodes - 1, 'started...', end='')
# #         for t in range(steps):

# #             #if render:
# #              #   env.render()

# #             #if t == 1: env.render()


# #             # print("state")
# #             # print(observation_tem.shape)  #(20, 124)
# #             # print(observation_tem.shape[0]) # 4/20
# #             # print(observation.shape[1]) #124

# #             # tem_len = observation_tem.shape[0]
# #             # tem_wid = observation_tem.shape[1]
# #             # tem_siz = tem_len*tem_wid

# #             # observation = np.reshape(observation_tem, (tem_siz, ))

# #             # print("observe")
# #             # print(observation.shape) # (2480, )

# #             # exit()
# #             print('SJF2: Episode', ep, 'step', t, '/', steps, 'started...')

# #             tem_done_s = False


# #             while not tem_done_s:

# #                # done = []
# #                 count_s += 1

# #                 action_s = env_s.get_action() #agent.act(observation)


# #                 # deleted by zc 07/05  # data file
# #                 # data.set_action(action.tolist())

# #                 # data.set_state(observation.tolist())

# #                 prev_observation_s = observation_s

# #                 print("action")
# #                 # print(action.shape)
# #                 print(action_s)

# #                 print("action_end")

# #                 ### 07/16#####
# #     #            noise = max(epsilon, 0) * ouf(action[0], 0.0, 0.5, 0.5,ransta1)

# #     #            tem_action = int(action[0] + noise)
# #                 # observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)
# #     #            if epsilon> 0: epsilon -= 1.0/100
                
# #                 tem_action_s = int(action_s[0])
# #                 observation_s, reward_s, done_s, info_s = env_s.step(tem_action_s if len(action_s) == 1 else action_s)

# #             # deleted by zc 07/05  # data file
# #             # data.set_reward(reward)

# #                 tem = int(env_s.pa.num_nw/2+1)
# #                 conc_act = np.zeros(tem) #(env_s.pa.num_nw+1)
# #                 for i in range(tem+1): #(env_s.pa.num_nw+1):
# #                     if i == tem_action_s:
# #                         conc_act[i] = 1


# #                 print("done test")
# #                 print(reward_s)
# #                 print(done_s)
# #                 print(info_s)



# #                 episode_s = {'obs': prev_observation_s,
# #                             'action': conc_act, #action_s,
# #                             'reward': reward_s,
# #                             'obs2': observation_s,
# #                             'done': done_s,
# #                             #   'done1': done[1],
# #                             't': t}

# #                 agent.observe(episode_s)

# #                 # total_reward += reward

# #                 tem_done_s = info_s


# #                 if tem_done_s:
# #                     total_reward_s += reward_s


# #                 if done_s:
# #                     print("done")
# #                     print(done_s)

# #             if ep == episodes -1:
# #                 f=open("/home/veronica/Desktop/env/job_profile/0726/sjf2/lastepreward.txt", "a")
# #                 f.write("%s\n" % reward_s)
# #                 f.close() 

# #             if done_s or (t == steps - 1):
# #                 t += 1
# #                 reward_sum_s += total_reward_s
# #                 # time_passed = timer.get_time()
# #                 print("done:", done_s)
# #                 # print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
# #                 #                                                             time_passed, round(
# #                 #                                                                 time_passed / t),
# #                 #                                                             round(reward_sum / (ep + 1))))


# #                 f2=open("/home/veronica/Desktop/env/job_profile/0726/sjf2/eachreward.txt", "a")
# #                 f2.write("%s\n" % total_reward_s)
# #                 f2.close()    


# #                 f1=open("/home/veronica/Desktop/env/job_profile/0726/sjf2/totalreward.txt", "a")
# #                 f1.write("%s\n" % reward_sum_s)
# #                 f1.close()                    

# #                 # deleted by zc 07/05  # data file
# #                 # data.finish_and_store_episode()


# #             #    if ep == 1 and done: exit()


# #                 break

# #     for ep in range(episodes):

# #         timer.reset()
# #         # observation_tem = env.reset()
# #         observation = env.reset()

# # #        env.jobsave()

# # #        tem_f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len.txt"
# #  #       tem_f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size.txt"
# # #
# # #        f1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len" + str(ep) + ".txt"
# # #        f2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size" + str(ep) + ".txt"

# # #        shutil.copy(tem_f1, f_1)
# # #        shutil.copy(tem_f2, f_2)

# #         total_reward = 0
# #         count = 0
# #         print('RL: Episode ', ep, '/', episodes - 1, 'started...', end='')
# #         for t in range(steps):

# #             #if render:
# #              #   env.render()

# #             #if t == 1: env.render()


# #             # print("state")
# #             # print(observation_tem.shape)  #(20, 124)
# #             # print(observation_tem.shape[0]) # 4/20
# #             # print(observation.shape[1]) #124

# #             # tem_len = observation_tem.shape[0]
# #             # tem_wid = observation_tem.shape[1]
# #             # tem_siz = tem_len*tem_wid

# #             # observation = np.reshape(observation_tem, (tem_siz, ))

# #             # print("observe")
# #             # print(observation.shape) # (2480, )

# #             # exit()
# #             print('RL2: Episode', ep, 'step', t, '/', steps, 'started...')

# #             tem_done = False


# #             while not tem_done:

# #                # done = []
# #                 count += 1

# #                 action = agent.act(observation)


# #                 # deleted by zc 07/05  # data file
# #                 # data.set_action(action.tolist())

# #                 # data.set_state(observation.tolist())

# #                 prev_observation = observation
# #                 print("action")
# #                 # print(action.shape)
# #                 print(action)
                

# #                 ### 07/16#####
# #     #            noise = max(epsilon, 0) * ouf(action[0], 0.0, 0.5, 0.5,ransta1)

# #     #            tem_action = int(action[0] + noise)
# #                 # observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)
# #     #            if epsilon> 0: epsilon -= 1.0/100
                
# #                 tem_action = np.argmax(action) #int(action[0])
# #                 print(tem_action)
# #                 print("action_end")
# #                 observation, reward, done, info = env.step(tem_action) # if len(action) == 1 else action)

# #             # deleted by zc 07/05  # data file
# #             # data.set_reward(reward)


# #                 print("done test")
# #                 print(reward)
# #                 print(done)
# #                 print(info)



# #                 episode = {'obs': prev_observation,
# #                             'action': action,
# #                             'reward': reward,
# #                             'obs2': observation,
# #                             'done': done,
# #                             #   'done1': done[1],
# #                             't': t}

# #                 agent.observe(episode)

# #                 # total_reward += reward

# #                 tem_done = info


# #                 if tem_done:
# #                     total_reward += reward


# #                 if done:
# #                     print("done")
# #                     print(done)

# #             if ep == episodes -1:
# #                 f=open("/home/veronica/Desktop/env/job_profile/0726/normal2/lastepreward.txt", "a")
# #                 f.write("%s\n" % reward)
# #                 f.close() 

# #             if done or (t == steps - 1):
# #                 t += 1
# #                 reward_sum += total_reward
# #                 time_passed = timer.get_time()
# #                 print("done:", done)
# #                 print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
# #                                                                             time_passed, round(
# #                                                                                 time_passed / t),
# #                                                                             round(reward_sum / (ep + 1))))


# #                 f2=open("/home/veronica/Desktop/env/job_profile/0726/normal2/eachreward.txt", "a")
# #                 f2.write("%s\n" % total_reward)
# #                 f2.close()    


# #                 f1=open("/home/veronica/Desktop/env/job_profile/0726/normal2/totalreward.txt", "a")
# #                 f1.write("%s\n" % reward_sum)
# #                 f1.close()                    

# #                 # deleted by zc 07/05  # data file
# #                 # data.finish_and_store_episode()


# #             #    if ep == 1 and done: exit()


# #                 break


# #     # end of episodes
# #     time = full_epoch_timer.get_time()
# #     print('Run {} episodes in {} seconds and got {} average reward'.format(
# #         episodes, time / 1000, reward_sum / episodes))


#     # deleted by zc 07/05  # data file
#     # data.save()


#     # running: 
#     # timer.reset()
#     # observation_tem = env.reset()

    

    for ep in range(episodes):
        observation1 = env.reset()
        run_reward = 0
        count = 0

        for t1 in range(steps):

            print('Episode', ep, '/', episodes, 'step', t1, '/', steps, 'started...')

            tem_done1 = False

            while not tem_done1:

                count+=1
                action1 = agent.act(observation1)
                prev_observation1 = observation1
                print("action")
                # print(action.shape)
                print(action1)
            
                tem_action1 = np.argmax(action1) #int(action1[0])
                print(tem_action1)
                # observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)
                print("action_end")
                observation1, reward1, done1, info1 = env.step(tem_action1) # if len(action1) == 1 else action1)


            # run_reward += reward1

                tem_done1 = info1

                if tem_done1:
                    run_reward += reward1

            print("done:", done1)

            if ep == episodes -1:
                f3=open("/home/veronica/Desktop/env/job_profile/0726/normal/run_lastreward.txt", "a")
                f3.write("%s\n" % reward1)
                f3.close()  

            if done1 or (t1 == steps - 1):
                t1 += 1
                    
                f4=open("/home/veronica/Desktop/env/job_profile/0726/normal/run_reward.txt", "a")
                f4.write("%s\n" % run_reward)
                f4.close()


                tem_t_info1 = []
                tem_t_info1.append(t1)
                tem_t_info1.append(done1)

                f=open("/home/veronica/Desktop/env/job_profile/0726/normal/run_done_info.txt", "a")
                f.write("%s\n" % tem_t_info1)
                f.close()  

                break

    # observation = env.reset()  # 07/12  added later

    env.close()

def ouf(x, mu, theta, sigma, rand):
    noise = theta * (mu - x) + sigma * rand.randn(1)
    return noise


if __name__ == '__main__':
    # np.random.seed(2)
    np.set_printoptions(threshold=sys.maxsize)
    start = time.time()
    run()
    end = time.time()
    print("running time: ")
    print(end-start)
