from sklearn.metrics import classification_report
import torch


class Meter:
    def __init__(self):
        self.predictions = []
        self.labels = []

    def update(self, predictions, labels):
        if isinstance(predictions, torch.Tensor): predictions = predictions.data.cpu().numpy()
        if isinstance(labels, torch.Tensor): labels = labels.data.cpu().numpy()

        self.predictions.extend(predictions)
        self.labels.extend(labels)

    def reset(self):
        self.predictions, self.labels = [], []

    def get_report(self):
        return classification_report(self.labels, self.predictions, output_dict=True,
                                     labels=range(4), target_names=range(4))
