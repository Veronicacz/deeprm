import gym
from gym import spaces, utils #errors
from gym.utils import seeding
import numpy as np
# import gym.envs.gym_foo.job_distribution 
# import gym.envs.gym_foo.parameters as parameters
from gym.envs.gym_foo_ocot.job_distribution import *
from gym.envs.gym_foo_ocot.parameters import *
# import job_distribution
# import parameters
import math
import matplotlib.pyplot as plt
import shutil


class FooocEnv(gym.Env):
	"""docstring for FooEnv"""

	metadata = {'render.modes': ['human']}

	def __init__(self, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42,   render=False, 
                 repre='image', end='all_done'):  # 'no_new_job'
		# self.pa = parameters.Parameters()
		self.ransta = np.random.RandomState(123)
		self.pa = Parameters(self.ransta)
		self.observation_space = spaces.Box(0, 255, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype = np.int16)
		# self.action_space = spaces.Box(0, 1, shape = (self.pa.num_nw,), dtype = np.int16)
		self.action_space = spaces.Discrete(self.pa.num_nw+1) #, shape=(1,), dtype = np.int16)
		# self.action_space.low = np.array([0])
		# self.action_space.high = np.array([self.action_space.n])
		self.nw_dist = self.pa.dist.bi_model_dist
		self.curr_time = 0
		self.end = end


		# print("network_input_height")
		# print(self.pa.network_input_height)  # 20
		# print("network_input_width")
		# print(self.pa.network_input_width)  #124

		# #set up random seed
		# if self.pa.unseen:
		# 	np.random.seed(314159)
		# else:
		# 	np.random.seed(seed)

		if nw_len_seqs is None or nw_size_seqs is None:
			# generate new work
			self.nw_len_seqs, self.nw_size_seqs = \
            	self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
			# self.workload = np.zeros(self.pa.num_res)
			# for i in range(self.pa.num_res):
			# 	self.workload[i] = \
   #                  np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
   #                  float(self.pa.res_slot) / \
   #                  float(len(self.nw_len_seqs))
			# 	print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
			self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
			self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
		else:
			self.nw_len_seqs = nw_len_seqs
			self.nw_size_seqs = nw_size_seqs
		self.seq_no = 0  # which example sequence
		self.seq_idx = -1  # index in that sequence
		# initialize system
		self.machine = Machine(self.pa, self.ransta)
		self.job_slot = JobSlot(self.pa)
		self.job_backlog = JobBacklog(self.pa)
		self.job_record = JobRecord()
		self.extra_info = ExtraInfo(self.pa)

		self.disslog = Dismisslog(self.pa)
		self.usedcolormap = []
		self.usedcolormap.append(0)
		self.state = self.observe()


	def generate_sequence_work(self, joblist_len):
		nw_len_seq = np.zeros(joblist_len, dtype=int)
		nw_size_seq = np.zeros((joblist_len, self.pa.num_res), dtype=int)

		for i in range(joblist_len):

			# if np.random.rand() < self.pa.new_job_rate:  # a new job comes
			if self.ransta.rand() < self.pa.new_job_rate:
				nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()
		return nw_len_seq, nw_size_seq

	def get_new_job_from_seq(self, seq_no, seq_idx):
		new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],  # [self.pa.num_ex, self.pa.simu_len]
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
		return new_job

	def observe(self):
		backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
		image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

		ir_pt = 0 # represnts ir_pt_th columns (res cubes) in one row 

		# print("backlog current size")
		# print(self.job_backlog.curr_size)  # 1

		for i in range(self.pa.num_res):


            # for res 1:  res 1 cluster + num_nw jobs' res 1 sizes
            # canvas definition: self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot)) 
			image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
			ir_pt += self.pa.res_slot
			
			for j in range(self.pa.num_nw):
				if self.job_slot.slot[j] is not None:  # fill in a block of work   # self.job_slot.slot = [None] * pa.num_nw


###########################delete 07/10########################33
					# tem_color = np.unique(self.machine.canvas[:])
					# tem_usecolor = np.array(self.usedcolormap)
					# occup_color = np.append(tem_color, tem_usecolor)
					# # WARNING: there should be enough colors in the color map
					# for color in self.machine.colormap:
					# 	if color not in occup_color:
					# 		n_color = color
					# 		break
					image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1 # n_color
					#########################################above#########################
				ir_pt += self.pa.max_job_size

			# print("test")
			# print(self.job_backlog.curr_size)  # 1

		# ###########modified by zc###############
		# if self.job_backlog.curr_size == 0:
		# 	image_repr[0,ir_pt: ir_pt + backlog_width] = 250 # color 250 is always assigned to backlog 

		# else:
		# 	print("backlog size")
		# 	print(self.job_backlog.curr_size)
		# 	print(backlog_width)
		# 	print(1/3)
		# 	print(self.job_backlog.curr_size / backlog_width)
		# 	image_repr[0: self.job_backlog.curr_size / backlog_width,ir_pt: ir_pt + backlog_width] = 250
		# ########################################

		image_repr[: int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + backlog_width] = 1 # 250

		if self.job_backlog.curr_size % backlog_width > 0: # the last line of backlog is not complete
			image_repr[int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1 # 250
		ir_pt += backlog_width

		# image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
  #                                             float(self.extra_info.max_tracking_time_since_last_job)

		image_repr[:, ir_pt: ir_pt + 1] = self.disslog.size
		ir_pt += 1

		assert ir_pt == image_repr.shape[1]

		# print("observation: ")
		# print(image_repr)

		return image_repr

	
	#def plot_state(self):
	# 	plt.figure("screen", figsize=(20, 5))

	# 	skip_row = 0

	 #	tem_obv = self.observe()


	 #	plt.imshow(tem_obv, interpolation='nearest', vmax=250)


	 #	plt.show()

	# 	for i in range(self.pa.num_res):

	# 		plt.subplot(self.pa.num_res,
    #                     1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                     i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
			
	# 		plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=250 #1)

	# 		for j in range(self.pa.num_nw):

	# 			job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))

	# 			if self.job_slot.slot[j] is not None:  # fill in a block of work

	# 				job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

	# 			plt.subplot(self.pa.num_res,
 #                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

	# 			plt.imshow(job_slot, interpolation='nearest', vmax=1)

	# 			if j == self.pa.num_nw - 1:
	# 				skip_row += 1

	# 	skip_row -= 1
	# 	backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
	# 	backlog = np.zeros((self.pa.time_horizon, backlog_width))

	# 	# backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
	# 	backlog[: int(self.job_backlog.curr_size / backlog_width), : backlog_width] = 1

	# 	# backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1
	# 	tem_ra = (self.job_backlog.curr_size % backlog_width)
	# 	backlog[int(self.job_backlog.curr_size / backlog_width), : tem_ra] = 1

	# 	plt.subplot(self.pa.num_res,
 #                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                    self.pa.num_nw + 1 + 1)

	# 	plt.imshow(backlog, interpolation='nearest', vmax=1)

	# 	plt.subplot(self.pa.num_res,
 #                    1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
 #                    self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

	# 	extra_info = np.ones((self.pa.time_horizon, 1)) * \
 #                     self.extra_info.time_since_last_new_job / \
 #                     float(self.extra_info.max_tracking_time_since_last_job)

	# 	plt.imshow(extra_info, interpolation='nearest', vmax=1)

	# 	plt.show()     # manual
 #        # plt.pause(0.01)  # automatic

	def get_reward(self):

		reward = 0
		for j in self.machine.running_job:  # running
			reward += self.pa.delay_penalty / float(j.len)

		for j in self.job_slot.slot:  # job queue
			if j is not None:
				reward += self.pa.hold_penalty / float(j.len)

		for j in self.job_backlog.backlog:  # backlog
			if j is not None:
				reward += self.pa.dismiss_penalty / float(j.len)

		for j in self.disslog.log:  # drop
			if j is not None:
				reward += self.pa.drop_penalty / float(j.len)

		return reward

	def step(self, a, repeat=False):
		status = None

#		done = []
#		done.append(False)
#		done.append(False)

		done = False
		reward = 0
		info = None
		#info = False

		#print("act")
		#print(a)
		# if a == self.pa.num_nw:  # explicit void action
		if a == 0: # no job is assigned
			status = 'MoveOn'
		elif self.job_slot.slot[a-1] is None:  # implicit void action
			status = 'MoveOn'
		else:
            # a is 0 --- num_nw, index is 0 --- num_nw - 1
			allocated = self.machine.allocate_job(self.job_slot.slot[a-1], self.curr_time)
			if not allocated:  # implicit void action
				status = 'MoveOn'
			else:
				status = 'Allocate'

		if status == 'MoveOn':
			#print("move on")
			self.curr_time += 1
			self.machine.time_proceed(self.curr_time)
			self.extra_info.time_proceed()

			info = True

            # add new jobs
			self.seq_idx += 1

###############07/07##################			

			if self.end == "no_new_job":  # end of new job sequence # different "end" standard
				if self.seq_idx >= self.pa.simu_len:
					done = True
			elif self.end == "all_done":  # everything has to be finished
				if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
					done = True


###########above####################
			#	elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
			#		done = True

					# modified 0705
			# reward = self.get_reward()  # reward update in each moving time steps 
##############modified##############
			if not done:

				if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
					new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)
					# print("new job profile")
					# print(new_job)
				else: new_job = None

				if new_job != None and new_job.len > 0:  # a new job comes, it is possible new_job.len = 0 (>arrival rate)

					print("new job comes")
							
					to_backlog = True
					# print("job slot")
					# print(self.job_slot.slot[0])
					# print(type(self.job_slot.slot[0]))
					# if self.job_slot.slot[3] == None:
					# 	print("yes")
					# else: exit()
					for i in range(self.pa.num_nw):
						if self.job_slot.slot[i] is None:  # put in new visible job slots
							self.job_slot.slot[i] = new_job
							self.job_record.record[new_job.id] = new_job
							to_backlog = False
							break

					# print(to_backlog)
					if to_backlog:
						if self.job_backlog.curr_size < self.pa.backlog_size:
							self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
							self.job_backlog.curr_size += 1
							self.job_record.record[new_job.id] = new_job
						else:  # abort, backlog full
							print("Backlog is full.")
							self.disslog.log[self.disslog.size] = new_job
							self.disslog.size += 1
                              # exit(1)
					self.extra_info.new_job_comes()
##############modified#####################
			# reward = self.get_reward()

		elif status == 'Allocate': # still needs to check another job in the queue, no job comes yet


#################07/07#########################
			# time still processes  # current schedule, current process immediately
			self.curr_time += 1
			self.machine.time_proceed(self.curr_time)
			self.extra_info.time_proceed()

			# info = False

			self.job_record.record[self.job_slot.slot[a-1].id] = self.job_slot.slot[a-1]
			self.job_slot.slot[a-1] = None

            # dequeue backlog
			if self.job_backlog.curr_size > 0: 
				self.job_slot.slot[a-1] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
				self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
				self.job_backlog.backlog[-1] = None
				self.job_backlog.curr_size -= 1


###########07/07##############
###################07/05####################################
			# add new jobs
			self.seq_idx += 1


			if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
				new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)
				# print("new job profile")
				# print(new_job)
			else: new_job = None

			if new_job != None and new_job.len > 0:  # a new job comes, it is possible new_job.len = 0 (>arrival rate)

				print("new job comes")
							
				to_backlog = True
				# print("job slot")
				# print(self.job_slot.slot[0])
				# print(type(self.job_slot.slot[0]))
				# if self.job_slot.slot[3] == None:
				# 	print("yes")
				# else: exit()
				for i in range(self.pa.num_nw):
					if self.job_slot.slot[i] is None:  # put in new visible job slots
						self.job_slot.slot[i] = new_job
						self.job_record.record[new_job.id] = new_job
						to_backlog = False
						break

				# print(to_backlog)
				if to_backlog:
					if self.job_backlog.curr_size < self.pa.backlog_size:
						self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
						self.job_backlog.curr_size += 1
						self.job_record.record[new_job.id] = new_job
					else:  # abort, backlog full
						print("Backlog is full.")
						self.disslog.log[self.disslog.size] = new_job
						self.disslog.size += 1
                            # exit(1)
				self.extra_info.new_job_comes()

		# modified 0705
		reward = self.get_reward()  # reward update in each time step

################################above################################################

		ob = self.observe()

		info = self.job_record  # modify 07/05
		# info = status

#		if done: # the jobs in the first seq_no (from seq_idx = 0 to seq_no = simu_len) have all been checked,
#			self.seq_idx = 0   #  [seq_no, seq_idx],  <--> [self.pa.num_ex, self.pa.simu_len]


#			if not repeat:
#				self.seq_no = (self.seq_no + 1) % self.pa.num_ex

#			self.reset()

		#if self.render:
		#	self.plot_state()

		print("status:")
		print(status)
		# print(done)
		# print(info)
		print("reward:")
		print(reward)

		return ob, reward, done, info


#	def render(self,  mode='human'):
#		self.plot_state()



	def reset(self):


		###### 07/12 ############
		# f=open("/home/veronica/Desktop/env/job_profile/0710/normal/disslog.txt", "a")
#		tem_str1 = "nw_len," + self.nw_len_seqs + "\n" 
		# for item in self.nw_len_seqs:
		# 	f.write("%s\n" % self.disslog.log)
		# f.close()
###################################33333

		tem_seed = np.random.randint(9999)
		self.ransta = np.random.RandomState(tem_seed)

		############modified by 07/10#####################
#		self.pa = Parameters(self.ransta)
#		self.observation_space = spaces.Box(0, 255, shape=(self.pa.network_input_height, self.pa.network_input_width), dtype = np.int16)
#		self.action_space = spaces.Box(0, 1, shape = (self.pa.num_nw,), dtype = np.int16)
#		# self.action_space = spaces.Box(0, self.pa.num_nw-1, shape=(1,), dtype = np.int16)

#		self.nw_dist = self.pa.dist.bi_model_dist

#		nw_len_seqs=None
#		nw_size_seqs=None
#
#
#		if nw_len_seqs is None or nw_size_seqs is None:
#			# generate new work
#			self.nw_len_seqs, self.nw_size_seqs = \
#				self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
#			# self.workload = np.zeros(self.pa.num_res)
#			# for i in range(self.pa.num_res):
#			# 	self.workload[i] = \
#			# 		np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
#			# 		float(self.pa.res_slot) / \
#			# 		float(len(self.nw_len_seqs))
#			# 	print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
#			self.nw_len_seqs = np.reshape(self.nw_len_seqs,
#                                           [self.pa.num_ex, self.pa.simu_len])
#			self.nw_size_seqs = np.reshape(self.nw_size_seqs,
#                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
#		else:
#			self.nw_len_seqs = nw_len_seqs
#			self.nw_size_seqs = nw_size_seqs


		f1=open("/home/veronica/Desktop/env/job_profile/0711/normal/nwlen.txt", "a")
#		tem_str1 = "nw_len," + self.nw_len_seqs + "\n" 
		for item in self.nw_len_seqs:
			f1.write("%s\n" % item)
		f1.close()
#		f1.write(tem_str1)
#		f1.close()

		f2=open("/home/veronica/Desktop/env/job_profile/0711/normal/nwsize.txt", "a")
#		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
		for item in self.nw_size_seqs:
			f2.write("%s\n" % item)
#		f2.write(tem_str2)
		f2.close()

		total_load = 0
		for seqt in range(self.pa.simu_len):
			tem_load = self.nw_len_seqs[self.pa.num_ex-1][seqt] * self.nw_size_seqs[self.pa.num_ex-1][seqt][0]
			total_load += tem_load

		nord = (self.pa.episode_max_length + self.pa.time_horizon) * self.pa.res_slot
		workload = total_load / float(nord)

		f3=open("/home/veronica/Desktop/env/job_profile/0711/normal/workload.txt", "a")
#		tem_str2 = "nw_size, " + self.nw_size_seqs + "\n"
		f3.write("%s\n" % workload)
#		f2.write(tem_str2)
		f3.close()		

		# self.end='no_new_job'
		self.end = 'all_done'

		self.curr_time = 0

        # initialize system
		self.machine = Machine(self.pa, self.ransta)
		self.job_slot = JobSlot(self.pa)
		self.job_backlog = JobBacklog(self.pa)
		self.job_record = JobRecord()
		self.extra_info = ExtraInfo(self.pa)
		self.disslog = Dismisslog(self.pa)

		# print("size")
		# print(self.job_backlog.curr_size)  # 0


		self.seq_no = 0  # which example sequence
		self.seq_idx = -1  # index in that sequence

		self.usedcolormap = []
		self.usedcolormap.append(0)

		self.state = self.observe()

#		self.jobsave()

		return self.state

	# def render(self, mode='human', close=False):
	# 	if not close:
 #            raise NotImplementedError(
 #                "This environment does not support rendering")


#	def jobsave(self):
#		filename1 = "/home/veronica/Desktop/env/job_file/smallcase/nw_len.txt" #+ str(i)
#		np.savetxt(filename1, self.nw_len_seqs)
#		filename2 = "/home/veronica/Desktop/env/job_file/smallcase/nw_size.txt" #+ str(i)
#		np.savetxt(filename2, self.nw_size_seqs)


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    def __init__(self, pa):
       self.slot = [None] * pa.num_nw
       # self.slot = np.zeros(pa.num_nw, dtype=int)


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Dismisslog:
	def __init__(self, pa):
		self.log = [None] * pa.episode_max_length
		self.size = 0


class Machine:
    def __init__(self, pa, ransta):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        self.ransta = ransta

        # shape = (time_horizon, num_res), each value = res_slot
        # available res slots for each resource at each time
        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot  

        self.running_job = []

        # colormap for graphical representation
        # self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap)) # !! 
        self.colormap = np.arange(5, 201, 2)  # pixel value, from 1 to 255, it cannot be 0, as canvas's initial value is 0
        # np.random.shuffle(self.colormap)
        self.ransta.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, curr_time):

        allocated = False

        for t in range(0, self.time_horizon - job.len): # in the graph, curr_time = first line in time_horizon = (t = 0)

            # to check if res has enough res slots to execute jobs
            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True #  job can be assigned

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time # in the graph, = t
                canvas_end_time = job.finish_time - curr_time # in the graph, = t + job_length

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):

                        # return index
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0] # those slots with 0 value are not occupied

                        # occupy from first ==0 canvas slot, to job.res_vec[res]_th
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot # add a new line (all slots are available) to the end of the avbl_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0  # # add a new line (no color) to the end of the canvas


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1

