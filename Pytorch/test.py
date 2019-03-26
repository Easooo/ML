#author：easooo
#date：2019.3

import torch
import numpy as np
x = torch.ones(3,5,requires_grad=True,dtype=torch.float)
size = x.size()[1:]
num_flag = 1
for s in size:
    num_flag *= s
tmp = x.view(-1,num_flag)
# print(tmp)
target = torch.ones(6,dtype=torch.float)  # 随机值作为样例
target1 = target.view(2, -1)
target2 = target.view(-1,2)
print(target)
print(target1)
print(target2)

a = np.ones(5)
a1 = a.reshape(1,-1)
a2 = a.reshape(-1,1)
print(a)
print(a1)
print(a2)
# y = x + 2
# z = y * y * 3
# out = z.mean()
# print(z)
# out.backward()
# # z.backward(torch.tensor([[0.1,1.0],[0.001,0.0001]],dtype=torch.float))
# print(x.grad)
 