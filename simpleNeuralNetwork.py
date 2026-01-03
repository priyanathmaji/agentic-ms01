import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNeuralNetwork():
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,1))
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        self.n_epoch = 100


    def createModel(self,x: torch.tensor,y: torch.tensor):
        for epoch in range(self.n_epoch):
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            print("Loss:", loss)
            self.optimizer.step()

    def infer(self,x) -> torch.tensor:
        output = self.model(x)

    def data_small(self):
        x = torch.rand(1000, dtype=torch.float)
        x = x.unsqueeze(0).T
        print(x.shape)
        #print(x)
        y = x ** 2
        return x, y
        
    def main(self):

        x, y = self.data_small()
        self.createModel(x,y)

if __name__ == "__main__":
    SimpleNeuralNetwork().main()
    x = torch.tensor([.020])
    y = SimpleNeuralNetwork().model(x)
    print(y)

