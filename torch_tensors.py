import torch


#1-d tensor
one_D = torch.tensor([2,2,1,1])
print('1-D Tensor ->>> 1 D Array')
print(one_D)

#2-d tensor
two_D = torch.tensor([[2,1,4,],[3,5,4],[1,2,0]])
print('2-D tensor ->>> Matrix, 2d array')
print(two_D)

# shap of tensors
print('1d size')
print(one_D.shape)
print(one_D.size())

print('2d size')
print(two_D.shape)
print(two_D.size())


#height/ num rows
print('num rows')
print(two_D.shape[0])

#float tensor
float_tensor = torch.FloatTensor([[2,1.9,4,],[3,5,4],[1,2,0]])
print('Tensor with real values')
print(float_tensor)


#mean
print('mean of tensor with real values')
print(float_tensor.mean())

#std
print('std of tensor with real values')
print(float_tensor.std())

# reshape tensor with tensor.view(num row, num col)
print('reshape 2d')
print(two_D.view(-1,1))


# 3d tensor (channels,rows,columns)
three_D = torch.randn(2,3,4)
print('3d')
print(three_D)
