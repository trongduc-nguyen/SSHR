import torch

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, lr_power=0.9):
        # Preserve the released recipe: the scalar wt_dec is SGD momentum;
        # per-parameter-group weight_decay values remain defined by the caller.
        super().__init__(params, lr=lr, momentum=weight_decay, weight_decay=0.0)

        self.global_step = 0
        self.max_step = max_step
        self.lr_power = lr_power

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.lr_power

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1








