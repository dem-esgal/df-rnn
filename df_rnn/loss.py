import torch
from torch import nn

from df_rnn.trx_encoder import PaddedBatch


def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    device = pred.device
    return torch.mean(torch.sum(-soft_targets.to(device) * logsoftmax(pred), 1))


def kl(pred, soft_targets):
    eps = 1e-7
    softmax = torch.nn.Softmax(dim=1)
    device = pred.device
    return torch.mean(
        torch.sum(soft_targets.to(device) * torch.log(soft_targets.to(device) / (softmax(pred) + eps) + eps), 1))


def mse_loss(pred, actual):
    device = pred.device
    return torch.mean((pred - actual.to(device)) ** 2)


def mape_metric(pred, actual):
    eps = 1
    device = pred.device
    return torch.mean((actual.to(device) - pred).abs() / (actual.to(device).abs() + eps))


def r_squared(pred, actual):
    device = pred.device
    return 1 - torch.sum((actual.to(device) - pred) ** 2) \
           / torch.sum((actual.to(device) - torch.mean(actual.to(device))) ** 2)


class AllStateLoss(nn.Module):
    def __init__(self, point_loss):
        super().__init__()
        self.loss = point_loss

    def forward(self, pred: PaddedBatch, true):
        y = torch.cat([torch.Tensor([yb] * length) for yb, length in zip(true, pred.seq_lens)])
        weights = torch.cat([torch.arange(1, length + 1) / length for length in pred.seq_lens])

        loss = self.loss(pred, y, weights)

        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, true):
        return self.loss(pred.float(), true.float())


class PseudoLabeledLoss(nn.Module):
    def __init__(self, loss, pl_threshold=0.5, unlabeled_weight=1.):
        super().__init__()
        self.loss = loss
        self.pl_threshold = pl_threshold
        self.unlabeled_weight = unlabeled_weight

    def forward(self, pred, true):
        label_pred, unlabel_pred = pred['labeled'], pred['unlabeled']
        if isinstance(self.loss, nn.NLLLoss):
            pseudo_labels = torch.argmax(unlabel_pred.detach(), 1)
        elif isinstance(self.loss, BCELoss):
            pseudo_labels = (unlabel_pred.detach() > 0.5).type(torch.int64)
        else:
            raise Exception(f'unknown loss type: {self.loss}')

        # mask pseudo_labels, with confidence > pl_threshold
        if isinstance(self.loss, nn.NLLLoss):
            probs = torch.exp(unlabel_pred.detach())
            mask = (probs.max(1)[0] > self.pl_threshold)
        elif isinstance(self.loss, BCELoss):
            probs = unlabel_pred.detach()
            mask = abs(probs - (1 - pseudo_labels)) > self.pl_threshold
        else:
            mask = torch.ones(unlabel_pred.shape[0]).bool()

        Lloss = self.loss(label_pred, true)
        if mask.sum() == 0:
            return Lloss
        else:
            Uloss = self.unlabeled_weight * self.loss(unlabel_pred[mask], pseudo_labels[mask])
            return (Lloss + Uloss) / (1 + self.unlabeled_weight)


def get_loss(params):
    loss_type = params['train.loss']

    if loss_type == 'bce':
        loss = BCELoss()
    elif loss_type == 'NLLLoss':
        loss = nn.NLLLoss()
    elif loss_type == 'mae':
        loss = nn.L1Loss()
    elif loss_type == 'mse':
        loss = MSELoss()
    elif loss_type == 'pseudo_labeled':
        loss = PseudoLabeledLoss(
            loss=get_loss(params['labeled']),
            pl_threshold=params['pl_threshold'],
            unlabeled_weight=params['unlabeled_weight']
        )
    else:
        raise Exception(f'unknown loss type: {loss_type}')

    if params.get('head', {}).get('pred_all_states_loss', False):
        loss = AllStateLoss(loss)

    return loss
