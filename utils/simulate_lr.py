'''
Visualises how the learning rate is changed by scheduler. 
'''

import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Dummy model parameters
params = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

T0 = 10
Tmult = 1
# Optimizer and scheduler
optimizer = AdamW(params, lr=1e-4, weight_decay=5e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tmult)

# Simulate 32 epochs with steps per epoch
epochs = 32
steps_per_epoch = 1  # step once per epoch for visualization
lrs = []

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

# Plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs * steps_per_epoch + 1), lrs, marker='o')
plt.title(f"CosineAnnealingWarmRestarts (T0={T0}, Tmult={Tmult}) over 32 epochs")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()
