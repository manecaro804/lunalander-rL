import torch
import numpy as np
from dqn_cnn import DQNCNN

# 🔥 Simular estado correcto (4 frames)
state = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)

# Convertir a formato PyTorch
state = np.transpose(state, (2, 0, 1))  # (4,84,84)
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1,4,84,84)

model = DQNCNN(n_actions=9)

output = model(state)

print("Output shape:", output.shape)