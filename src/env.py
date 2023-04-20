import numpy as np
import torch
import random
from collections import deque

class Swarm:
    def __init__(self, n_drones, grid_size, init_state):
        self.n_drones = n_drones
        self.positions = []
        self.state = init_state
        pos = 0
        for i in range(len(init_state[0])):
            if init_state[0][i] == 2:
                self.positions.append(i)
        self.grid_size = grid_size
        self.state_size = self.grid_size[0] * self.grid_size[1]
        self.action_size = self.n_drones
        self.coverage_area = set()

    def step(self, actions):
        rewards = torch.zeros(self.n_drones)
        
        cur_state = self.state.clone()
        # print("Taking step:")
        for i, action in enumerate(actions):

            pos = self.positions[i]
            row = pos // self.grid_size[1]
            col = pos % self.grid_size[1]

            
            cur_state[0][pos] = max(cur_state[0][pos] - 2, 1)  # covered
            row_old, col_old = row, col
            
            if action == 0: # move up
                row = max(0, row - 1)
            elif action == 1: # move down
                row = min(self.grid_size[0] - 1, row + 1)
            elif action == 2: # move left
                col = max(0, col - 1)
            elif action == 3: # move right
                col = min(self.grid_size[1] - 1, col + 1)
            
            self.positions[i] = row*self.grid_size[1] + col
            if cur_state[0][self.positions[i]] == 1:
                cur_state[0][self.positions[i]] = 2
            else:
                cur_state[0][self.positions[i]] += 2
            if self.positions[i] not in self.coverage_area:
                self.coverage_area.add(self.positions[i])
                rewards[i] = 1
            if row == row_old and col == col_old:
                rewards[i] = -1
                # print(f"Unchanged {row}, {col}, Action: {action}")
                # print(row*self.grid_size[1] + col)
        self.state = cur_state
        done = len(self.coverage_area) == self.state_size
        return rewards, done
    
    def get_state(self):
        return self.state
    
    def reset(self, init_state):
        self.state = init_state
        self.coverage_area = set()
        self.positions = []
        pos = 0
        for i in range(len(init_state[0])):
            if init_state[0][i] == 2:
                self.positions.append(i)
