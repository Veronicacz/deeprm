import numpy as np 



print(1/3)


b = np.random.randint(5777)
print(b)


# c = np.empty([2, 2])
# print(c[0])

# if c[0][0] == None: print("succ")
# else: print("Fail")

c = np.array([2,2])
c = c[:, None]
print(c)

if c[0][0] == 2: print("succ")