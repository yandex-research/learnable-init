import torch
import torch.nn as nn
import contextlib
import uuid

BATCHNORM_STATE = {f'cuda:{i}': None for i in range(torch.cuda.device_count())}

def reset_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
        elif isinstance(module, ContextualBatchNorm2d):
            raise Exception("'reset_batchnorm' is called for models containing only nn.BatchNorm")


@contextlib.contextmanager
def track_batchnorm_stats(current_stats, updated_stats=None, device='cuda:0'):
    global BATCHNORM_STATE
    device = 'cuda:0' if device == 'cuda' else device
    prev_state = BATCHNORM_STATE[device]
    BATCHNORM_STATE[device] = current_stats, updated_stats
    try:
        yield
    finally:
        BATCHNORM_STATE[device] = prev_state


def initialize_batchnorm_stats(model):
    stats = {}
    for module in model.modules():
        if isinstance(module, ContextualBatchNorm2d):
            stats[module.uid] = module.get_init_running_stats()
    return stats


@contextlib.contextmanager
def activate_context_batchnorm(model):
    def forward_change(model):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_features = module.weight.shape[0]
                device = module.weight.device
                setattr(model, name, ContextualBatchNorm2d(num_features).to(device=device))
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise Exception("Now only BatchNorm2d are supported")
            elif len(list(module.children())) > 0:
                forward_change(module)

    def backward_change(model):
        for name, module in model.named_children():
            if isinstance(module, ContextualBatchNorm2d):
                num_features = module.weight.shape[1]
                device = module.weight.device
                setattr(model, name, nn.BatchNorm2d(num_features).to(device=device))
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise Exception("nn.BatchNorm cannot appear here")
            elif len(list(module.children())) > 0:
                backward_change(module)

    forward_change(model)
    try:
        yield
    finally:
        backward_change(model)


class ContextualBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.uid = str(uuid.uuid4())
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)

    def get_init_running_stats(self):
        return (
            torch.tensor(0, dtype=torch.long, device=self.weight.device),
            torch.zeros(1, self.num_features, 1, 1, device=self.weight.device),
            torch.ones(1, self.num_features, 1, 1, device=self.weight.device)
        )

    def forward(self, inputs):
        global BATCHNORM_STATE
        assert len(inputs.shape) == 4  # [b, c, h, w]
        device_id = self.weight.get_device()
        assert BATCHNORM_STATE[f"cuda:{device_id}"] is not None, "This layer can only be used inside track_batchnorm_stats context"
        current_stats, updated_stats = BATCHNORM_STATE[f"cuda:{device_id}"]
        running_mean, running_inv_std, num_batches_tracked = current_stats[self.uid]

        if self.training:
            batch_mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
            batch_inv_std = 1. / torch.sqrt(inputs.var(dim=(0, 2, 3), keepdim=True) + self.eps)

            input_normalized = (inputs - batch_mean) * (self.weight * batch_inv_std) + self.bias

            # update stats
            new_running_mean = (1 - self.momentum) * running_mean + self.momentum * batch_mean
            new_running_inv_std = (1 - self.momentum) * running_inv_std + self.momentum * batch_inv_std
            new_num_batches_tracked = num_batches_tracked + 1
            assert updated_stats is not None, "Please set updated_states param in track_batchnorm_stats context"
            assert self.uid not in updated_stats, \
                "This layer was already used in the current context. We do not support sharing batchnorms yet."
            updated_stats[self.uid] = new_running_mean, new_running_inv_std, new_num_batches_tracked
        else:
            input_normalized = (inputs - running_mean) * (self.weight * running_inv_std) + self.bias

        return input_normalized