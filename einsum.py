import torch

a = torch.tensor([[1,2,3],[3,0,3]])
print('input a', a)
#Permutation of Tensor
b = torch.einsum('ij->ji',a)
print('ij->ji', b)
#Summation
b = torch.einsum('ij->', a)
print('ij->', b)
b = torch.einsum('ij->j', a)
print('ij->j', b)
b = torch.einsum('ij->i', a)
print('ij->i', b)

v = torch.tensor([[1,3,4]])
b = torch.einsum('ij,kj->ik', a,v)
print('v', v)
print('a.shape', a.shape)
print('v.shape', v.shape)
print('b.shape', b.shape)

#Matrix, Matrix multiplication, (proper multiplication)
b = torch.einsum('ij,kj->ik', a,a)
print('ij,kj->ik',b)

#Dot product forst row with first row of matrix
b = torch.einsum('i,i->', a[0], a[0])
print('i,i->',b)

#Dot product with matrix 
b = torch.einsum('ij,ij->',a,a)
print('ij,ij->',b)

#Hadamard Product (elementwise multiplication)
b = torch.einsum('ij,ij->ij',a,a)
print('ij,ij->ij', b)
