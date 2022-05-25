import torch

from df_rnn.lightning_modules.module_base import ModuleBase
from torch.nn import BCELoss
from df_rnn.seq_to_target import EpochAuroc


class ClsModule(ModuleBase):
    """pl.LightningModule for training Classification embeddings

    Parameters
    ----------
    seq_encoder : torch.nn.Module
        sequence encoder
    head : torch.nn.Module
        head of th model
    loss : torch.nn.Module
        Loss function for training
    lr : float. Default: 1e-3
        Learning rate
    weight_decay: float. Default: 0.0
        weight decay for optimizer
    lr_scheduler_step_size : int. Default: 100
        Period of learning rate decay.
    lr_scheduler_step_gamma: float. Default: 0.1
        Multiplicative factor of learning rate decay.
    """

    def __init__(
            self,
            seq_encoder: torch.nn.Module,
            head: torch.nn.Module,
            loss: torch.nn.Module = None,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            lr_scheduler_step_size: int = 50,
            lr_scheduler_step_gamma: float = 0.5):
        train_params = {
            'train.lr': lr,
            'train.weight_decay': weight_decay,
            'lr_scheduler': {
                'step_size': lr_scheduler_step_size,
                'step_gamma': lr_scheduler_step_gamma
            }
        }

        super().__init__(train_params, seq_encoder, loss)

        self._head = head

    @property
    def metric_name(self):
        return 'valid_auroc'

    @property
    def is_requires_reduced_sequence(self):
        return False

    def get_loss(self):
        return BCELoss()

    def get_validation_metric(self):
        return EpochAuroc()

    def shared_step(self, x, y):
        y_h = self.seq_encoder(x)
        y_h = self._head(y_h)
        return y_h, y

    def predict_step(self, batch, _):
        y_h = self.seq_encoder(batch)
        return self._head(y_h)
