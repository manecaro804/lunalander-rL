import gymnasium as gym
import ale_py
import torch
import numpy as np

from wrappers import PreprocessFrame, FrameStack
from dqn_cnn import DQNCNN

gym.register_envs(ale_py)


def play():
    env = gym.make("ALE/Pacman-v5", render_mode="human")
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)

    model = DQNCNN(n_actions=env.action_space.n)
    model.load_state_dict(torch.load("results/dqn_pacman_final.pth", map_location="cpu"))
    model.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_t = np.transpose(state, (2, 0, 1))
        state_t = torch.tensor(state_t, dtype=torch.float32).unsqueeze(0) / 255.0

        with torch.no_grad():
            action = model(state_t).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward

    print(f"Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    play()