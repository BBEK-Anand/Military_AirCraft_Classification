

import torch
from PTLF.utils import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score
)


class Acc(Metric):
    def __init__(self):
        super().__init__()

    def _setup(self, args):
        self.accuracy = MulticlassAccuracy(num_classes=74)
        
    def forward(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=1)  # Convert logits to class indices
        acc = self.accuracy(y_pred, y_true)
        return acc.item()


class AUROC(Metric):
    def __init__(self):
        super().__init__()
        self.args = {'average'}

    def _setup(self, args):
        self.metric = MulticlassAUROC(num_classes=74, average=args['average'])

    def forward(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1) 
        return self.metric(probs, targets)


class AUPRC(Metric):
    def __init__(self):
        super().__init__()
        self.args = {'average'}

    def _setup(self, args):
        self.metric = MulticlassAveragePrecision(num_classes=74, average=args['average'])

    def forward(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        return self.metric(probs, targets)


class F1Score(Metric):
    def __init__(self):
        super().__init__()
        self.args = {'average'}
        
    def _setup(self, args):
        self.metric = MulticlassF1Score(num_classes=74, average=args['average'])

    def forward(self, outputs, targets):
        preds = torch.argmax(outputs, dim=1)
        return self.metric(preds, targets)
