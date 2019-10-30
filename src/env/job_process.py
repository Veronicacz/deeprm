import numpy as np
import math
# import job_running



f1 = open("nwlen.txt", 'r')
lines1=f1.readlines()
nw_len = []
#print(len(lines1))
len_len = len(lines1)
# print(len_len)
# print(len_len/3)

numl = 0 
tem_list = []
for line1 in lines1:
    tem_l1 = line1.replace('[','').replace(']','')  #.replace(' ',',')
    row = tem_l1.split()
    if numl % 10 == 0 and numl != 0:
        nw_len.append(tem_list)
        tem_list = []
    tem_save = []
    for item in row:
        tem_save.append(int(item))

    # nw_len.append(tem_list)
    tem_list.append(tem_save)

    if numl == len(lines1)/10 - 1:
        nw_len.append(tem_list)

    numl +=1 

# print(len(nw_len))
f1.close()


f2 = open("nwsize.txt", 'r')
lines2=f2.readlines()
nw_size = []
count = 0
epos_list = []
#print(len(lines2))
len_size = len(lines2)
# print(len_size)
for line2 in lines2:
    tem_l2 = line2.replace('[','').replace(']','')  #.replace(' ',',')
    row = tem_l2.split()
    if count % 10 == 0 and count != 0: 
        nw_size.append(epos_list)
        epos_list = []
    tem_list = []
    for item in row:
        tem_list.append(int(item))

    epos_list.append(tem_list)

    if count == len(lines1)/10 -1:
        nw_size.append(epos_list)

    count += 1
f2.close()

f3 = open("nworder.txt", 'r')
lines3=f3.readlines()
nw_order = []
count2 = 0
epos_order = []
#print(len(lines2))
len_order = len(lines3)
# print(len_size)
for line3 in lines3:
    tem_l3 = line3.replace('[','').replace(']','')  #.replace(' ',',')
    row = tem_l3.split()
    if count2 % 10 == 0 and count2 != 0: 
        nw_order.append(epos_order)
        epos_order = []
    tem_order = []
    for item in row:
        tem_order.append(int(item))

    epos_order.append(tem_order)

    if count2 == len(lines1)/10 -1:
        nw_order.append(epos_order)

    count2 += 1
f3.close()


f4 = open("nwtrans.txt", 'r')
lines4=f4.readlines()
nw_trans = []
count4 = 0
epos_trans = []
#print(len(lines2))
len_trans = len(lines4)
# print(len_size)
for line4 in lines4:
    tem_l4 = line4.replace('[','').replace(']','')  #.replace(' ',',')
    row = tem_l4.split()
    if count4 % 10 == 0 and count4 != 0: 
        nw_trans.append(epos_trans)
        epos_trans = []
    tem_trans = []
    for item in row:
        tem_trans.append(int(item))

    epos_trans.append(tem_trans)

    if count4 == len(lines1)/10 -1:
        nw_trans.append(epos_trans)

    count4 += 1
f4.close()


# print(count)
print(len(nw_len))
print(len(nw_size))
print(len(nw_order))
print(len(nw_trans))
# print(len(nw_size[100]))
# print(nw_len[0])
# print(nw_size[0])
# print(nw_len[2507])
# print(nw_size[2507])

assert len_len == len_size
assert len_len == len_order
assert len_trans == len_len

# epsi = 100
# run_size = nw_size[:-1]
tem_size = np.array(nw_size)
print(len(tem_size))
# run_len = nw_len[:-1]
tem_len = np.array(nw_len)
tem_order = np.array(nw_order)
tem_trans = np.array(nw_trans)
print(len(tem_trans))
print(tem_trans)
print(len(tem_trans[0]))

# print(len(tem_len[0]))
# SJF = job_running.SJF(epsi, tem_len, tem_size)
# SJF.run()

# print(nw_len)
# # print(nw_size)
# print(len(nw_len))
# print(len(nw_size))
