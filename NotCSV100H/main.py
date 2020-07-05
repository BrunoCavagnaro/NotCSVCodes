import gym
import numpy as np
import matplotlib.pyplot as plt
iters = 1000
env = gym.make("Taxi-v3")
env.reset()
acts = []
episo = 10000
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.7
tR = 0
rewards = []

for episode in range(0, episo):
    done = False
    G, reward = 0,0
    state = env.reset()

    while done != True:
            action = np.argmax(Q[state]) #1
            state2, reward, done, info = env.step(action) #2
            Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
            G += reward
            state = state2

            rewards.append(reward)

    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))
plt.plot(range(len(episo)), rewards)
plt.show()
env.close()