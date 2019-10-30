import numpy as np
import math


class Dist:

    def __init__(self, num_res, num_clu, max_nw_size, job_len, ranstat, transrate):

        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len
        self.num_clu = num_clu
        self.rans = ranstat

        self.transrate = transrate

        self.job_small_chance = 0.3 #5 # 0.8

        self.job_len_big_lower = 9 #job_len * 2 / 3
        self.job_len_big_upper = 9 #job_len

        self.job_len_small_lower = 8 #1
        self.job_len_small_upper = 8 #job_len / 3

        self.job_size_big_lower = max_nw_size #* 2 / 3
        self.job_size_big_upper = max_nw_size

        self.job_size_small_lower = 9 #6 #9#1
        self.job_size_small_upper = 9 #6 #9 #max_nw_size / 3

        self.trans_ratio = 0.5

    def normal_dist(self):

        nw_size = np.zeros(self.num_clu)

        for i in range(self.num_clu):
            nw_size[i] = self.rans.randint(0, self.max_nw_size + 1)  # if size = 0, no sub task on this cluster # occupy how many cubes


        count = 0
        for i in range(self.num_clu):
            if nw_size[i] != 0:
                count+=1

        # new sub tasks duration
        # nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension
        nw_len = np.zeros(self.num_clu)
        for i in range(self.num_clu):
            if nw_size[i] != 0:
                nw_len[i] = self.rans.randint(1, self.job_len + 1)

        subtask_order = np.zeros(self.num_clu)
        tem_clu_arrange = np.arange(1, count+1)
        np.random.shuffle(tem_clu_arrange)

        arange_index = 0
        for i in range(self.num_clu):
            if nw_size[i] != 0:
                subtask_order[i] = tem_clu_arrange[arange_index]
                arange_index += 1

        nw_trans = np.zeros(self.num_clu)

        for i in range(1, self.num_clu):
            tem_f = np.where(subtask_order == i)
            for j in range(i+1, self.num_clu):
                # tem_t = np.where(subtask_order == i+1)
                tem_t = np.where(subtask_order == j)
                if nw_size[tem_t] != 0:
                    break
            rate = self.transrate[tem_f, tem_t]
            if rate != 0 and nw_size[tem_f] != 0 and nw_size[tem_t] != 0:
                workload = nw_size[tem_f] * nw_len[tem_f] * self.trans_ratio
                tem_trans = float(workload/rate)
                trans = math.ceil(tem_trans)
                nw_trans[tem_f] = trans


        return nw_len, nw_size, subtask_order, nw_trans

    def bi_model_dist(self):

        nw_size = np.zeros(self.num_clu)



        # for i in range(self.num_clu):
        #     nw_size[i] = self.rans.randint(0, self.max_nw_size + 1) # if size = 0, no sub task on this cluster


        # count = 0
        # for i in range(self.num_clu):
        #     if nw_size[i] != 0:
        #         count+=1


        # print(nw_size)
        nw_len = np.zeros(self.num_clu)


        if np.random.rand() < self.job_small_chance:
            for i in range(self.num_clu):
                nw_len[i] = self.rans.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
                nw_size[i] = self.rans.randint(self.job_size_big_lower,
                                       self.job_size_big_upper + 1)
        else:
            for i in range(self.num_clu):
                nw_len[i] = self.rans.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)
                nw_size[i] = self.rans.randint(self.job_size_small_lower,
                                       self.job_size_small_upper + 1)


        count = 0
        for i in range(self.num_clu):
            if nw_size[i] != 0:
                count+=1

        # subtask_order = np.zeros(self.num_clu)
        # tem_clu_arrange = np.arange(1, self.num_clu+1)
        # np.random.shuffle(tem_clu_arrange)
        # for i in range(self.num_clu):
        #     subtask_order[i] = tem_clu_arrange[i]


        subtask_order = np.zeros(self.num_clu)
        tem_clu_arrange = np.arange(1, count+1)
        np.random.shuffle(tem_clu_arrange)

        arange_index = 0
        for i in range(self.num_clu):
            if nw_size[i] != 0:
                subtask_order[i] = tem_clu_arrange[arange_index]
                arange_index += 1

        # for i in range(self.num_clu):
        #     if nw_size[i] != 0:
        #         if np.random.rand() < self.job_small_chance:
        #             nw_len[i] = self.rans.randint(self.job_len_small_lower,
        #                                    self.job_len_small_upper + 1)
        #         else:
        #             nw_len[i] = self.rans.randint(self.job_len_big_lower,
        #                                    self.job_len_big_upper + 1)


        nw_trans = np.zeros(self.num_clu)

        for i in range(1, self.num_clu):
            tem_f = np.where(subtask_order == i)
            for j in range(i+1, self.num_clu+1):
                # tem_t = np.where(subtask_order == i+1)
                tem_t = np.where(subtask_order == j)
                if nw_size[tem_t] != 0:
                    break
            rate = self.transrate[tem_f, tem_t]
            # print(self.transrate.shape)
            # print(tem_f)
            # print(tem_t)
            # print(self.transrate[tem_f, tem_t])
            # print(rate)
            # print(nw_size[tem_f])
            # print(nw_size[tem_t])
            if rate != 0 and nw_size[tem_f] != 0 and nw_size[tem_t] != 0:
                workload = 1 #10 #nw_size[tem_f] * nw_len[tem_f] * self.trans_ratio
                tem_trans = float(workload/rate)
                trans = math.ceil(tem_trans)
                nw_trans[tem_f] = trans

        return nw_len,  nw_size, subtask_order, nw_trans 