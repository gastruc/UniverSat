import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Metric


class NoMetrics(Metric):
    """
    Computes no metrics or saves a batch of reconstruction to visualise them
    Args:
        save_reconstructs (bool): if True saves a batch of reconstructions
        modalities (list): list of modalities used
        save_dir (str): where to save reconstructions
    """

    def __init__(
        self,
        save_reconstructs: bool = False,
        modalities: list = [],
        save_dir: str = '',
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_recons = save_reconstructs
        self.modalities = modalities
        if self.save_recons:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.saves = {}
            for modality in self.modalities:
                self.saves[modality] = []
                self.saves['_'.join(['gt', modality])] = []

    def update(self, pred, gt):
        if self.save_recons:
            recons, _ = pred
            for modality in self.modalities:
                if modality == 'aerial':
                    preds = recons['_'.join(['reconstruct', modality])]
                    target = gt[modality][:, :, :300, :300]
                else:
                    preds, mask = recons['_'.join(['reconstruct', modality])]
                    target = gt[modality][mask[:, 0], mask[:, 1]]
                indice = torch.randint(0, len(preds), (1,)).item()
                self.saves[modality].append(preds[indice])
                self.saves['_'.join(['gt', modality])].append(target[indice])

    def compute(self):
        if self.save_recons:
            for key in self.saves.keys():
                for i, tensor in enumerate(self.saves[key]):
                    torch.save(tensor.cpu(), self.save_dir + key + str(i) + ".pt")
        return {}

    def reset(self):
        """Reset the metric state."""
        super().reset()

class MetricsSeg(Metric):
    """
    SegPangaea is a class for evaluating segmentation models using a confusion matrix approach.

    Attributes:
        num_classes (int): Number of classes in the segmentation task
        ignore_index (int): Index value to ignore when computing metrics
        confusion_matrix (torch.Tensor): Matrix of shape (num_classes, num_classes) to store predictions

    Methods:
        update(pred, gt):
            Updates the confusion matrix with new predictions and ground truth.
            Args:
                pred (torch.Tensor): Model predictions
                gt (dict): Dictionary containing ground truth labels under 'label' key

        compute():
            Computes various metrics from the accumulated confusion matrix.
            Returns:
                dict: Dictionary containing the following metrics:
                    - mIoU: Mean Intersection over Union across all classes
                    - mF1: Mean F1 score across all classes
                    - mAcc: Mean pixel accuracy
    """

    def __init__(self, num_classes, ignore_value, ignore_column=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        if ignore_column is not None:
            self.ignore_column = ignore_column
        self.add_state("confusion_matrix", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, pred, gt):
        label = gt['label'].flatten(1, 2).long()
        pred = torch.argmax(pred, dim=1).flatten(1, 2).long()
        valid_mask = label != self.ignore_value
        pred, target = pred[valid_mask], label[valid_mask]
        count = torch.bincount(
            (pred * self.num_classes + target), minlength=self.num_classes ** 2
        )
        self.confusion_matrix = self.confusion_matrix.to(pred.device)
        self.confusion_matrix += count.view(self.num_classes, self.num_classes)

    def compute(self):
        if self.ignore_column is not None:
            self.confusion_matrix[:, self.ignore_column] = 0
            self.confusion_matrix[self.ignore_column, :] = 0
        # Calculate IoU for each class
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6))

        # Calculate precision and recall for each class
        precision = intersection / (self.confusion_matrix.sum(dim=0) + 1e-6)
        recall = intersection / (self.confusion_matrix.sum(dim=1) + 1e-6)

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        num_classes = self.num_classes
        if self.ignore_column is not None:
            num_classes -= 1

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.sum().item() / num_classes
        mf1 = f1.sum().item() / num_classes
        macc = (intersection.sum() / (self.confusion_matrix.sum() + 1e-6)).item()

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "mIoU": miou,
            "mF1": mf1,
            "mAcc": macc,
        }

        return metrics

    def reset(self):
        """Reset the metric state."""
        super().reset()


class OutDiversity(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("variance", torch.tensor(0.0, dtype=torch.float64),dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
    def update(self, pred, batch):
        #pred['predicted_tokens'] BLC
        variance = torch.var(pred['predicted_tokens'], dim=1).mean(dim=-1).sum()
        if not torch.isnan(variance):
            self.variance += variance
            self.count += pred['predicted_tokens'].shape[0]


    def compute(self):
        variance = self.variance / self.count
        return {'variance': variance.item()}

class ClusterHistogramme(Metric):
    def __init__(self, num_classes=15, prefix=""):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("histogram", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.prefix = prefix

    def update(self, pred, gt):
        key = 'predicted_tokens' if 'predicted_tokens' in pred else 'target_logits'
        prediction = torch.argmax(pred[key], dim=-1).reshape(-1)
        onehot = torch.nn.functional.one_hot(prediction, num_classes=self.num_classes)

        histogramme = torch.sum(onehot, dim=0)
        self.histogram += histogramme

    def compute(self):
        histogram = self.histogram / torch.sum(self.histogram)
        return {f'{self.prefix}assignment_histogram': histogram}

class ClusterDensity(Metric):
    def __init__(self, num_classes=15, prefix=""):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("density", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.prefix = prefix

    def update(self, pred, gt):
        key = 'predicted_tokens' if 'predicted_tokens' in pred else 'target_logits'
        prediction = torch.softmax(pred[key], dim=-1).flatten(0, 1)
        density = torch.sum(prediction, dim=0)
        self.density += density
        return {f'{self.prefix}assignment_density': density}

    def compute(self):
        density = self.density / torch.sum(self.density)
        return {f'{self.prefix}assignment_density': density}

class SSL_metrics(Metric):
    def __init__(self, num_classes=None):
        super().__init__()
        self.number_of_classes = num_classes

        self.Out_diversity = OutDiversity()

        self.add_state("acc", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        if num_classes is not None:
            self.ClusterHistogrammeStudent = ClusterHistogramme(num_classes=num_classes, prefix="student_")
            self.ClusterHistogrammeTeatcher = ClusterHistogramme(num_classes=num_classes, prefix="teacher_")
            self.ClusterDensityStudent = ClusterDensity(num_classes=num_classes, prefix="student_")
            self.ClusterDensityTeatcher = ClusterDensity(num_classes=num_classes, prefix="teacher_")
            self.Accuracy = Accuracy(task="multiclass", num_classes=num_classes)


    def update(self, pred, gt):
        self.Out_diversity.update(pred, gt)

        # pred = pred["predicted_tokens"]
        # target = gt["target"]
        # bs, nt, d = pred.shape

        # # pred_mu = pred.mean(1, keepdims=True)
        # # pred_std = pred.std(1, keepdims=True)
        # # pred = (pred - pred_mu) / (pred_std + 1e-4)

        # pred = F.normalize(pred, p=2, dim=-1)
        # target = F.normalize(target, p=2, dim=-1)
        # scores = torch.einsum("npd,nqd->npq", pred, target)
        # labels = torch.arange(nt, dtype=torch.long, device=pred.device)[
        #         None
        #     ].repeat(
        #         bs, 1
        #     )  # BxNmodel.target_head.positionwise_sk=False

        # self.acc += torch.mean((scores.argmax(dim=-1) == labels).float())
        # self.count += 1

        if self.number_of_classes is not None:
            self.ClusterHistogrammeStudent.update(pred, gt)
            self.ClusterHistogrammeTeatcher.update(gt, pred)
            self.ClusterDensityStudent.update(pred, gt)
            self.ClusterDensityTeatcher.update(gt, pred)
            self.Accuracy.update(pred['predicted_tokens'].argmax(dim=-1), gt['target'].argmax(dim=-1))

    def compute(self):
        out = self.Out_diversity.compute()
        # out.update({'acc_mim': self.acc / self.count})
        if self.number_of_classes is not None:
            out.update(self.ClusterHistogrammeStudent.compute())
            out.update(self.ClusterHistogrammeTeatcher.compute())
            out.update(self.ClusterDensityStudent.compute())
            out.update(self.ClusterDensityTeatcher.compute())
            out.update({'assignment_acc':self.Accuracy.compute()})
        return out

    def reset(self):
        super().reset()
        self.Out_diversity.reset()
        if self.number_of_classes is not None:
            self.ClusterHistogrammeStudent.reset()
            self.ClusterHistogrammeTeatcher.reset()
            self.ClusterDensityStudent.reset()
            self.ClusterDensityTeatcher.reset()
            self.Accuracy.reset()
