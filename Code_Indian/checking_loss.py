import torch
a = torch.FloatTensor([[1,2,3,4],[5,6,7,8]])
print(a)

print("OLD CODE #######################################")
print("a[:][:2] is ", a[:][:2])
# Output
#tensor([[1., 2., 3., 4.],
#        [5., 6., 7., 8.]])
print("a[:][2:] is ", a[:][2:])  
# Output
# tensor([], size=(0, 4))

print("MY NEW CODE######################################")
print("a[:,:2] is ", a[:,:2])

#Output 
#tensor([[1., 2.],
#        [5., 6.]]))
print("a[:,2:] is ", a[:,2:])
# Output
#tensor([[3., 4.],
#        [7., 8.]])


