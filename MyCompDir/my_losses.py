
from PTLF.utils import Loss

from torch import nn

class CrossEn(Loss):
    def __init__(self):
        super().__init__()
    def _setup(self, args):
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,pred, target):
        return self.criterion(pred, target)