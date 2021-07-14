import os
from warnings import warn
import time
import numpy as np
import torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .utils.general_utils import switch_initializers_device


class Trainer(nn.Module):
    def __init__(self, maml: nn.Module,
                 meta_lr=0.001, meta_betas=(0.9, 0.997), meta_grad_clip=None,
                 exp_name=None, recovery_step=None, device='cuda'):
        """
        Training helper that trains the model to minimize loss in a supervised mode,
        computes metrics and does a few other tricks if you ask nicely
        :param experiment_name: a path where all logs and checkpoints are saved
        :param extra_attrs: dict {name: module} to be saved inside trainer via setattr
        """
        super().__init__()

        if exp_name is None:
            exp_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(*time.gmtime()[:6])
            print('using automatic experiment name: ' + exp_name)

        self.experiment_path = os.path.join('logs/', exp_name)
        self.writer = SummaryWriter(self.experiment_path, comment=exp_name)
        self.device = device

        self.maml = maml
        self.meta_optimizer = torch.optim.Adam(list(maml.initializers.parameters()),
                                               lr=meta_lr, betas=meta_betas)
        self.meta_grad_clip = meta_grad_clip
        self.total_steps = 0

        if recovery_step is not None:
            checkpoint = torch.load(os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(recovery_step)))
            self.maml.load_state_dict(checkpoint['maml_state_dict'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
            self.total_steps = checkpoint['iteration'] + 1

        # Switch device for untrained initializers
        self.maml.untrained_initializers = switch_initializers_device(self.maml.untrained_initializers, device)

    def record(self, *, prefix='', **metrics):
        """
        Computes and saves metrics into tensorboard
        :param prefix: common prefix for tensorboard
        :param metrics: key-value parameters forwarded into every metric
        :return: metrics (same as input)
        """
        if not (prefix == '' or prefix.endswith('/')):
            warn("It is recommended that prefix ends with slash(/) for readability")

        for key, value in metrics.items():
            assert np.shape(value) == (), "metric {} must be scalar, but got {}".format(key, np.shape(value))
            self.writer.add_scalar(prefix + str(key), value, self.total_steps)
        return metrics

    def train_on_batch(self, train_loader, valid_loader, prefix='train/', **kwargs):
        """ Performs a single gradient update and reports metrics """
        pass

    def evaluate_metrics(self, train_loader, test_loader, prefix='val/', **kwargs):
        """ Predicts and evaluates metrics over the entire dataset """
        pass

    def save_model(self):
        torch.save({
            'iteration': self.total_steps,
            'maml_state_dict': self.maml.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict()},
            os.path.join(self.experiment_path, "checkpoint_{}.pth".format(self.total_steps))
        )