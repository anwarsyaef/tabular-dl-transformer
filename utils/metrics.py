from typing import Optional, Dict, TypedDict, Any

import torch
from sklearn.metrics import classification_report
from torch import nn


class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class MetricsTracker:
    def __init__(self, device=None):
        self.y_true = torch.tensor([], dtype=torch.int, device=device)
        self.y_prob = torch.tensor([], device=device)
        self.y_pred = torch.tensor([], dtype=torch.int, device=device)
        self.n_correct = 0

        self.report_cache: Optional[Dict[str, Any]] = None

    def append(self, batch_labels: torch.IntTensor, batch_outputs: torch.Tensor):
        """
        :param batch_labels: [batch_size] actual class of the label
        :param batch_outputs: [batch_size x n_classes] the raw logits predicted by the model
        :return: this instance
        """
        batch_outputs = batch_outputs.detach()
        batch_labels = batch_labels.detach()

        is_multiclass = len(batch_outputs.size()) > 1
        batch_outputs = (
            nn.functional.softmax(batch_outputs, dim=-1) if is_multiclass else nn.functional.sigmoid(batch_outputs))

        self.y_true = torch.cat((self.y_true, batch_labels))
        self.y_prob = torch.cat((self.y_prob, batch_outputs))

        y_pred = batch_outputs.argmax(1) if is_multiclass else torch.round(batch_outputs).type(torch.int64)
        self.y_pred = torch.cat((self.y_pred, y_pred))

        self.n_correct += (batch_labels == y_pred).type(torch.int).sum().item()
        self.report_cache = None

        return self

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.y_true.size()[0]

    @property
    def report(self) -> Metrics:
        if self.y_true.size()[0] == 0:
            raise ValueError("Unable to provide metrics without data")
        elif self.report_cache is None:
            # calling .cpu().numpy() directly resulted in an error sometimes
            y_true = torch.cat((torch.tensor([], device='cpu'), self.y_true.cpu()))
            y_pred = torch.cat((torch.tensor([], device='cpu'), self.y_pred.cpu()))
            self.report_cache: Dict[str, Any] = classification_report(y_true.numpy(),
                                                                      y_pred.numpy(),
                                                                      output_dict=True,
                                                                      zero_division=0)

        return {
            'accuracy': self.report_cache["accuracy"],
            'precision': self.report_cache["macro avg"]["precision"],
            'recall': self.report_cache["macro avg"]["recall"],
            'f1_score': self.report_cache["macro avg"]["f1-score"],
        }
