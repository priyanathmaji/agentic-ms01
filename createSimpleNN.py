import torch
import torch.nn as nn
import torch.optim as optim

x = torch.rand(10)
x = x * 10
x = x.unsqueeze(-1)
#x = torch.tensor([[1],[2],[3]])
y = x ** 2

print(x.shape)

model = nn.Sequential(nn.Linear(1,100), nn.ReLU(), nn.Linear(100,1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(100000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_function(y, y_pred)
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    x = torch.tensor([[2.0]])
    print(model(x))