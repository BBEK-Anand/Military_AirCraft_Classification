


from PTLF.utils import Optimizer

#define your Optimizer functions here
###DEMO
import torch.optim as optim


class OptAdam(Optimizer):
    def __init__(self):
        super().__init__()
        self.args = {'model_parameters', 'learning_rate'}

    def setup(self,args):
        learning_rate = args.get('learning_rate')
        self.optimizer = optim.Adam(args['model_parameters'], lr=learning_rate)
        return self
    def step(self, **kwargs):
        # Step function to apply the gradients and update model parameters
        self.optimizer.step()

    def zero_grad(self):
        # Zero the gradients before the backward pass
        self.optimizer.zero_grad()
