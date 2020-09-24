import numpy as np
class ScheduledOptim():

    def __init__(self, opt, start_lr, d_model, warmup):
        self.opt = opt
        self.start_lr = start_lr
        self.d_model = d_model
        self.warmup = warmup
        self.steps = 0

    def step_lr(self):
        self.update_lr()
        self.opt.step()

    def update_lr(self):
        self.steps +=1
        lr = self.start_lr * self.get_lr()

        for params in self.opt.param_groups:
            params['lr'] = lr

    def get_lr(self):
        steps, warmup = self.steps, self.warmup
        return (self.d_model ** -0.5)* min(steps ** (-0.5), steps * warmup ** (-1.5))

    def zero_grad(self):
        self.opt.zero_grad()
