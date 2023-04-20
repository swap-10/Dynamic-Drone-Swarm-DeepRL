import torch
import numpy as np
from env import Swarm
from network import PolicyNetwork
from agent import RLAgent

if __name__ == "__main__":
    n_drones = 10
    grid_size = (10, 10)
    state_size = grid_size[0] * grid_size[1]
    action_size = 4
    state = torch.zeros((1, state_size))
    init_indices = np.random.choice(len(state[0]), size=n_drones, replace=False)
    state[0][init_indices] = 2  # initial positions of the drones
    env = Swarm(n_drones=n_drones, grid_size=grid_size, init_state=state)
    model = PolicyNetwork(n_drones, state_size=state_size, action_size=action_size)
    agent = RLAgent(n_drones=10, state_size=state_size, action_size=action_size, model=model)
    agent.learn(env, n_episodes=100)