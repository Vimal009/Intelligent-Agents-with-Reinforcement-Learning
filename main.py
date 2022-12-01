import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#Environment Testing with Random Actions
# Create a Variables env
env_name = "CartPole-v0"
env=gym.make(env_name)
for episode in range(1,11):
   score = 0
   state = env.reset()
   done = False
   while not done:
       env.render()
       action = env.action_space.sample()
       n_state, reward, done, info = env.step(action)
       score += reward
   print('episode', episode, 'score', score)
env.close()

# Model Training
env_name = "CartPole-v0"
env = gym.make(env_name)
DummyVecEnv([lambda: env])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("PPOModel")

# Model Testing
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()