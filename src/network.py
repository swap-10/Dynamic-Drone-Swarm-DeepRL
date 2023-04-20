import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, n_drones, state_size, action_size):
        super().__init__()
        self.n_drones = n_drones
        self.x1 = nn.Linear(state_size, 64)
        self.x2 = nn.Linear(64, 32)
        self.x3 = nn.Linear(32, 32)
        self.x4 = nn.Linear(32, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, state):
        op = self.x1(state)
        op = self.relu(op)
        op = self.x2(op)
        op = self.relu(op)
        op = self.x3(op)
        op = self.relu(op)
        op = self.x4(op)
        op = self.softmax(op)
        outputs = []
        for i in range(self.n_drones):
            outputs.append(op)
        return outputs