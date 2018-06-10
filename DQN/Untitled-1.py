import gym
env = gym.envs.make('Breakout-v0')
 
env.reset()
for t in range(1000):
    observation =env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, _= env.step(action)
    if t % 100 == 0:
        env.reset()