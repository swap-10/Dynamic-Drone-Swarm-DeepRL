import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

all_states = []

class RLAgent:
    def __init__(self, n_drones, state_size, action_size, model, learning_rate=0.001, discount_factor=0.95, batch_size=32, memory_size=10000):
        self.n_drones = n_drones
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.model = model
        self.tensorboard_writer  = SummaryWriter('runs/drones_rl_expt_1')
        


    def act(self, state):
        act_values = self.model(state)
        # if np.random.rand() <= self.epsilon:
        #    return
        act_values = [Categorical(act) for act in act_values]
        
        return act_values
    
    def plot_state(self, state, nrows, ncols):
        for i in range(nrows):
            for j in range(ncols):
                print(f"{state[0][i*ncols+j]}", end=" | ")
            print("\n" + "-"*60)
        print("Pure state: ", state)
        print("\n\n\n\n\n")
    
    # Currently unused
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Currently unused
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        actions = np.zeros((self.batch_size, self.n_drones), dtype=int)
        rewards = np.zeros((self.batch_size, self.n_drones))
        next_states = np.zeros((self.batch_size, self.state_size))
        dones = np.zeros((self.batch_size,))
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
            
        
        targets = self.model.predict(states)
        q_next = self.model.predict(next_states)
        for i in range(self.batch_size):
            for j in range(self.n_drones):
                if dones[i]:
                    targets[i][j] = rewards[i][j]
                else:
                    targets[i][j] = rewards[i][j] + self.discount_factor * np.max(q_next[i][j])
                    print(f"Done with {i}, {j}")
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, env, n_episodes):
        torch.autograd.set_detect_anomaly(True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        for i_episode in range(n_episodes):
            
            state = torch.zeros((1, self.state_size))
            init_indices = np.random.choice(len(state[0]), size=self.n_drones, replace=False)
            state[0][init_indices] = 2  # initial positions of the drones

            env.reset(state)
            done = False
            total_reward = 0
            timestep = 0
            all_states.append([])

            while not done and timestep < 1000:
                actions = self.act(env.get_state())
                # act_values = [torch.argmax(ac, dim=1) for ac in actions]
                act_values = [act.sample() for act in actions]
                rewards, done = env.step(act_values)
                
                loss = [actions[i].log_prob(act_values[i])*(rewards[i]+ (1e-2)) for i in range(self.n_drones)]
                loss = -1 * sum(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_reward = len(env.coverage_area)
                
                # state = next_state
                if (timestep % 100 == 0 or timestep % 100 == 1):
                    print(f"Timestep: {timestep}, Rewards: {total_reward}")
                    self.plot_state(env.get_state(), 10, 10)
                    
                all_states[-1].append(np.reshape(env.get_state(), env.grid_size))
                timestep += 1
                self.tensorboard_writer.add_scalar(f'Reward progression {i_episode}', total_reward, timestep)
                # next_state = np.array([len(env.coverage_area) / (env.grid_size[0] * env.grid_size[1])])
                # self.remember(state, actions, rewards, next_state, done)
                # self.replay()
            print("Episode {}/{}, Total Reward: {}, Epsilon: {:.2}, Timesteps: {}"
                  .format(i_episode + 1, n_episodes, total_reward, self.epsilon, timestep))
            self.plot_state(env.get_state(), 10, 10)
