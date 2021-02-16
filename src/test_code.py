import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# a = torch.zeros(4,5,6)
# index = 0
# for i in range(4):
# 	for j in range(5):
# 		a[i,j:] = index
# 		index+=1

# b = copy.deepcopy(a)
# b = b.permute(1,0,2)
# print(b)

######################################

c = torch.zeros(4,5,6)
index__ = 0
for i in range(4):
	for j in range(5):
		for k in range(6):
			c[i,j,k] = index__
			index__+=1
print(c)
c = c.permute(1,0,2)

c = c[2:]
c = c.permute(1,0,2)
print(c)