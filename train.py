import torch
import random
from env import SoccerEnv
from model import DQN

env = SoccerEnv()

state_size = 9
action_size = 6

model = DQN(state_size, action_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

gamma = 0.99

for episode in range(5000):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    total_reward = 0

    for step in range(500):
        if random.random() < 0.1:
            action = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                q = model(state)
                action = torch.argmax(q).item()

        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Q-learning update
        q_values = model(state)
        q_value = q_values[action]

        with torch.no_grad():
            next_q = model(next_state).max()
            target = reward + gamma * next_q * (1 - done)

        loss = (q_value - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

        if done:
            break

    print("Episode:", episode, "Reward:", total_reward)
