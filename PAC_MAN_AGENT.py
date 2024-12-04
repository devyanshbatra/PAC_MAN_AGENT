import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# Custom Maze Environment
class MazeEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.maze = np.zeros((self.size, self.size))
        self.player_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.maze[tuple(self.goal_pos)] = 2
        return self._get_state()

    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        move = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
        new_pos = [max(0, min(self.size-1, self.player_pos[i] + move[i])) for i in range(2)]
        
        reward = -1  # Small penalty for each move
        done = False

        if new_pos == self.goal_pos:
            reward = 100  # Large reward for reaching the goal
            done = True

        self.player_pos = new_pos
        return self._get_state(), reward, done

    def _get_state(self):
        state = self.maze.copy()
        state[tuple(self.player_pos)] = 1
        return state

# Neural Network
class MazeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MazeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_dim * input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Agent
class MazeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 1e-3
        
        self.policy_net = MazeNet(state_size, action_size).to(self.device)
        self.target_net = MazeNet(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randrange(self.action_size)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.from_numpy(np.array(states)).float().unsqueeze(1).to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        
        # Double DQN
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update of the target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop
env = MazeEnvironment(size=10)
agent = MazeAgent(env.size, 4)
n_episodes = 1000
max_steps = 200
scores = []

for e in range(n_episodes):
    state = env.reset()
    score = 0
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        agent.replay()
        if done:
            break
    scores.append(score)
    print(f"Episode: {e+1}/{n_episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

# Visualization
plt.plot(scores)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()

# Custom visualization of the trained agent
def visualize_agent(agent, env, max_steps=100):
    state = env.reset()
    done = False
    steps = 0
    
    plt.figure(figsize=(8, 8))
    while not done and steps < max_steps:
        plt.clf()
        plt.imshow(state, cmap='coolwarm')
        plt.title(f'Step {steps}')
        plt.pause(0.1)
        
        action = agent.act(state)
        state, _, done = env.step(action)
        steps += 1
    
    plt.clf()
    plt.imshow(state, cmap='coolwarm')
    plt.title(f'Final State (Steps: {steps})')
    plt.show()

visualize_agent(agent, env)