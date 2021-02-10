import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


a = torch.zeros(4,5,6)
index = 0
for i in range(4):
	for j in range(5):
		a[i,j:] = index
		index+=1
print(a)
print(a.view(1,1,-1))