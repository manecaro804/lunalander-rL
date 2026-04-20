import gymnasium as gym
import ale_py
from wrappers import PreprocessFrame, FrameSkip

gym.register_envs(ale_py)

env = gym.make("ALE/Pacman-v5", render_mode="human")
env = FrameSkip(env, skip=4)
env = PreprocessFrame(env)

obs, _ = env.reset()

print(obs.shape)


for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)

    if done or truncated:
        obs, _ = env.reset()

env.close()