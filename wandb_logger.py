import wandb

class WandbLogger:
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb

    def log(self, metrics):
        if self.use_wandb:
            wandb.log(metrics)
