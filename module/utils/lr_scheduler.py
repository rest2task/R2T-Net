import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warm-up and restarts.

    - Warm-up: linearly increase LR from base_lr to max_lr over `warmup_steps`.
    - Cosine decay: after warm-up, decay LR from max_lr to min_lr following a half cosine curve.
    - Restarts: once down to min_lr, optionally restart cycle with decayed max_lr.

    Args:
        optimizer: Wrapped optimizer.
        first_cycle_steps (int): Total steps in the first cosine cycle (including warm-up).
        max_lr (float): Peak learning rate.
        min_lr (float): Minimum learning rate at end of cycle.
        warmup_steps (int): Number of initial steps for warm-up.
        gamma (float): Multiplicative factor to decay max_lr at each restart.
        last_epoch (int): The index of last epoch.
    """
    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        max_lr,
        min_lr=1e-6,
        warmup_steps=0,
        gamma=1.0,
        last_epoch=-1
    ):
        self.first_cycle_steps = first_cycle_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        # Internal state
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # During warm-up
        if self.step_in_cycle < self.warmup_steps:
            return [
                base_lr + (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                for base_lr in self.base_lrs
            ]

        # Cosine decay after warm-up
        progress = (self.step_in_cycle - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps)
        return [
            self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) * (self.gamma ** self.cycle)
            for _ in self.base_lrs
        ]

    def step(self, epoch=None):
        # Restart cycle if exceeded
        if self.step_in_cycle >= self.first_cycle_steps:
            self.cycle += 1
            # Reset step within cycle (if epoch given, use it)
            self.step_in_cycle = epoch - 1 if epoch is not None else -1

        self.step_in_cycle += 1
        super().step(epoch)
