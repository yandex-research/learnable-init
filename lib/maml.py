from collections import namedtuple
from itertools import chain
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimizers import IngraphGradientDescent
from .utils.general_utils import copy_and_replace, do_not_copy, nested_flatten, nested_pack, NONE_TENSOR, is_none_tensor
from .context_batchnorm import track_batchnorm_stats, initialize_batchnorm_stats
from .models.fixup_resnet import FixupResNet
from .initializers import *
from .plif_initializers import *
from torch.utils.checkpoint import checkpoint


INITIALIZED_MODULE_TYPES = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)
INITIALIZED_RECURRENT_MODULE_TYPES = (nn.LSTMCell, nn.GRUCell)


class MAML(nn.Module):
    Result = namedtuple('Result', ['model', 'train_loss_history', 'valid_loss_history', 'optimizer_state'])

    def __init__(self, model: nn.Module, model_type: str,
                 loss_function=F.cross_entropy,
                 optimizer=IngraphGradientDescent(0.01), checkpoint_steps=1):
        """ Module that attempts to change model by performing SGD (with optional momentum and rms scaling)
            :param model: a torch module that will be edited
            :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
                By default this function should be non-negative and loss == 0 is a trigger to finish editing
            :param optimizer: in-graph optimizer that creates updated copies of model
            :param checkpoint_steps: uses gradient checkpoints every *this many* steps
                    Note: this parameter highly affects the memory footprint
        """
        super().__init__()
        self.model, self.loss_function, self.optimizer = model, loss_function, optimizer
        self.initializers = nn.ModuleDict()
        self.checkpoint_steps = checkpoint_steps
        self.model_type = model_type
        self.key_name2name = {}

        for name, module in self.model.named_modules():
            weight_initializer = None
            bias_initializer = None

            if isinstance(module, INITIALIZED_MODULE_TYPES):
                if model_type == 'AE':
                    weight_initializer = normal_initializer_from_weight(module.weight)
                elif model_type == 'lstm':
                    # Initialize logits layer by LM initializer
                    weight_initializer = normal_initializer_from_weight(module.weight)
                elif isinstance(model, FixupResNet):
                    weight_initializer = fixup_resnet_module_weight_initializer(name, module, model.num_layers)
                else:
                    weight_initializer = normal_initializer_from_weight(module.weight)

                if module.bias is not None:
                    if model_type in 'AE':
                        bias_initializer = kaiming_normal_bias_initializer(module.weight)
                    elif model_type == 'lstm':
                        # Initialize logits layer by LM initializer
                        bias_initializer = normal_initializer_from_weight(module.bias)
                    elif isinstance(model, FixupResNet):
                        bias_initializer = None
                    else:
                        raise Exception("Unknown model type")

            elif isinstance(module, INITIALIZED_RECURRENT_MODULE_TYPES):
                num_gates = 4 if isinstance(module, nn.LSTMCell) else 3
                hidden_size = module.hidden_size
                weight_initializer = nn.ModuleDict()
                for weight_name, weights in module._parameters.items():
                    if weight_name == 'bias_hh': continue
                    for gate_id in range(num_gates):
                        gate_name = '_'.join((weight_name, 'gate_{}'.format(gate_id)))

                        initializer = kaiming_normal_initializer_given_fan(hidden_size)
                        weight_initializer[gate_name] = initializer

            elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
                weight_initializer = normal_initializer(std=1.0)
                assert name not in self.initializers.keys(), 'model has to have unique layer names'

            key_name = "_".join(name.split("."))
            assert key_name not in self.initializers.keys(), 'model has to have unique layer names'
            if weight_initializer is None and bias_initializer is None:
                if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                    print(name)
            else:
                self.initializers[key_name] = nn.ModuleList([weight_initializer, bias_initializer])
                self.key_name2name[key_name] = name

        self.untrained_initializers = {key: deepcopy(value) for key, value in self.initializers.items()}

    @staticmethod
    def get_parameters(model):
        for name, module in model.named_modules():
            if isinstance(module, INITIALIZED_MODULE_TYPES):
                yield module.weight
                if module.bias is not None:
                    yield module.bias
            elif isinstance(module, INITIALIZED_RECURRENT_MODULE_TYPES):
                for weight_name in ['weight_ih', 'weight_hh', 'bias_ih']:
                    yield module._parameters[weight_name]
            else:
                if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                    yield from module.parameters()

    def resample_parameters(self, initializers=None, is_final=False):
        # Reset stats for nn.BatchNorm2d
        for module in self.model.modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm)):
                assert is_final, "nn.BatchNorm is allowed only in final evaluation mode"
                module.reset_running_stats()

        initializers = initializers or self.initializers
        for key_name, (weight_initializers, bias_initializers) in initializers.items():
            name = self.key_name2name[key_name]
            module = dict(self.model.named_modules())[name]

            if isinstance(module, INITIALIZED_RECURRENT_MODULE_TYPES):
                num_gates = 4 if isinstance(module, nn.LSTMCell) else 3
                hidden_size = module.hidden_size
                assert bias_initializers is None, 'LSTM quantile functions are not splited yet'

                # ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']
                for weight_name, init_weights in module._parameters.items():
                    if weight_name.startswith('bias_hh'):
                        module._parameters[weight_name] = nn.Parameter(torch.zeros_like(init_weights),
                                                                       requires_grad=False)
                        continue

                    weights = []
                    for gate_id in range(num_gates):
                        gate_weights = init_weights[gate_id * hidden_size: (gate_id + 1) * hidden_size]
                        gate_name = '_'.join((weight_name, 'gate_{}'.format(gate_id)))

                        initializer = weight_initializers[gate_name]
                        weights.append(initializer(torch.rand_like(gate_weights)))

                    weights = torch.cat(weights, dim=0)
                    if is_final:
                        module._parameters[weight_name] = nn.Parameter(weights, requires_grad=True)
                    else:
                        module._parameters[weight_name] = weights

            elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
                weights = weight_initializers(torch.rand_like(module.weight))
                module._parameters['weight'] = weights if not is_final else nn.Parameter(weights, requires_grad=True)

            else:
                weights = weight_initializers(torch.rand_like(module.weight))
                module._parameters['weight'] = weights if not is_final else nn.Parameter(weights, requires_grad=True)

                if module.bias is not None and bias_initializers is not None:
                    bias = bias_initializers(torch.rand_like(module.bias))
                    module._parameters['bias'] = bias if not is_final else nn.Parameter(bias, requires_grad=True)

    def forward(self, inputs, targets, valid_inputs, valid_targets,
                first_valid_step=10, valid_loss_interval=10,
                opt_kwargs=None, loss_kwargs=None, optimizer_state=None, device='cuda', **kwargs):
        """
        Apply optimizer to the model (out-of-place) and return an updated copy
        :param inputs: data that is fed into the model
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param optimizer_state: if specified, the optimizer starts with this state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: updated_model, loss_history, optimizer_state
            * updated_model: a copy of model that was trained for len(inputs) steps, differentiable w.r.t. original
            * loss_history: a list of loss function values BEFORE each optimizer update; differentiable
            * optimizer_state: final state of the chosen optimizer AFTER the last step; you guessed it, differentiable
        :rtype: MAML.Result
        """
        assert isinstance(inputs, (list, tuple)) and \
               isinstance(targets, (list, tuple)), 'Train inputs are required to be lists or tuples'
        assert isinstance(valid_inputs, (list, tuple)) and \
               isinstance(valid_targets, (list, tuple)), 'Valid inputs are required to be lists or tuples'
        assert len(inputs) == len(targets) > 0, "Non-empty inputs are required"

        opt_kwargs, loss_kwargs = opt_kwargs or {}, loss_kwargs or {}
        parameters_to_copy = list(self.get_parameters(self.model))
        parameters_not_to_copy = [param for param in chain(self.model.parameters(), self.model.buffers())
                                  if param not in set(parameters_to_copy)]

        if optimizer_state is None:
            optimizer_state = self.optimizer.get_initial_state(self.model, parameters=parameters_to_copy, **opt_kwargs)

        # initial maml state
        initial_step_index = torch.zeros(1, requires_grad=True)
        initial_batchnorm_stats = initialize_batchnorm_stats(self.model)
        initial_maml_state = (initial_step_index, parameters_to_copy, optimizer_state, initial_batchnorm_stats)
        flat_maml_state = list(nested_flatten(initial_maml_state))

        # WARNING: this code treats parameters_to_copy and parameters_not_to_copy as global
        # variables for _maml_internal. Please DO NOT change or delete them in this function
        def _maml_internal(_steps, *_flat_maml_state):
            step_index, trainable_parameters, _optimizer_state, current_batchnorm_stats = \
                nested_pack(_flat_maml_state, structure=initial_maml_state)
            updated_model = copy_and_replace(
                self.model, dict(zip(parameters_to_copy, trainable_parameters)), parameters_not_to_copy)

            is_first_pass = not torch.is_grad_enabled()
            # Note: since we use gradient checkpoining, this code will be executed two times:
            # (1) initial forward with torch.no_grad(), used to create checkpoints
            # (2) second forward with torch.enable_grad() used to backpropagate from those checkpoints
            # During first pass, we deliberately set detach=True to avoid creating inter-checkpoint graph

            inner_valid_losses, inner_train_losses = [], []
            for _ in range(int(_steps)):
                updated_batchnorm_stats = {}
                with torch.enable_grad(), do_not_copy(*parameters_not_to_copy), \
                        track_batchnorm_stats(current_batchnorm_stats, updated_batchnorm_stats, device=device):

                    predictions = updated_model(inputs[int(step_index)].to(device))
                    train_loss = self.loss_function(predictions, targets[int(step_index)].to(device), **loss_kwargs)
                    inner_train_losses.append(train_loss)
                    _optimizer_state, updated_model = self.optimizer.step(
                        _optimizer_state, updated_model, loss=train_loss, detach=is_first_pass,
                        parameters=self.get_parameters(updated_model), **kwargs)

                    current_batchnorm_stats = updated_batchnorm_stats

                step_index = step_index + 1
                if int(step_index) >= first_valid_step and \
                   int(step_index) % valid_loss_interval == 0:
                    valid_step_index = (int(step_index) - first_valid_step) // valid_loss_interval
                    with track_batchnorm_stats(current_batchnorm_stats, device=device):
                        updated_model.eval()  # Turns ContextualBatchNorm and Dropout in eval mode
                        valid_predictions = updated_model(valid_inputs[valid_step_index].to(device))
                        valid_loss = self.loss_function(valid_predictions, valid_targets[valid_step_index].to(device),
                                                        **loss_kwargs)
                        inner_valid_losses.append(valid_loss)
                        updated_model.train()

            new_maml_state = (step_index, list(self.get_parameters(updated_model)),
                              _optimizer_state, current_batchnorm_stats)
            inner_train_losses = torch.stack(inner_train_losses) if len(inner_train_losses) > 0 else NONE_TENSOR
            inner_valid_losses = torch.stack(inner_valid_losses) if len(inner_valid_losses) > 0 else NONE_TENSOR

            outputs = (inner_train_losses, inner_valid_losses, *nested_flatten(new_maml_state))
            return tuple(tensor if tensor.requires_grad else tensor.clone().requires_grad_(True)
                         for tensor in outputs)

        train_loss_history = []
        valid_loss_history = []

        for chunk_start in range(0, len(inputs), self.checkpoint_steps):
            steps = min(self.checkpoint_steps, len(inputs) - chunk_start)
            train_losses, valid_losses, *flat_maml_state = \
                checkpoint(_maml_internal, torch.as_tensor(steps), *flat_maml_state)

            if not is_none_tensor(train_losses):
                train_loss_history.extend(train_losses.split(1))
            if not is_none_tensor(valid_losses):
                valid_loss_history.extend(valid_losses.split(1))

        _, final_trainable_parameters, final_optimizer_state, batchnorm_stats = \
            nested_pack(flat_maml_state, structure=initial_maml_state)
        final_model = copy_and_replace(
            self.model, dict(zip(parameters_to_copy, final_trainable_parameters)), parameters_not_to_copy)
        return self.Result(final_model,
                           train_loss_history=train_loss_history,
                           valid_loss_history=valid_loss_history,
                           optimizer_state=final_optimizer_state)


class PLIF_MAML(MAML):
    Result = namedtuple('Result', ['model', 'train_loss_history', 'valid_loss_history', 'optimizer_state'])

    def __init__(self, model: nn.Module, model_type: str,
                 loss_function=F.cross_entropy,
                 optimizer=IngraphGradientDescent(0.01), checkpoint_steps=1, num_quantiles=100):
        """ Module that attempts to change model by performing SGD (with optional momentum and rms scaling)
            :param model: a torch module that will be edited
            :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
                By default this function should be non-negative and loss == 0 is a trigger to finish editing
            :param optimizer: in-graph optimizer that creates updated copies of model
            :param checkpoint_steps: uses gradient checkpoints every *this many* steps
                    Note: this parameter highly affects the memory footprint
        """
        super().__init__(model, model_type, loss_function, optimizer, checkpoint_steps)
        self.model, self.loss_function, self.optimizer = model, loss_function, optimizer
        self.initializers = nn.ModuleDict()
        self.key_name2name = {}
        self.num_quantiles = num_quantiles

        for name, module in self.model.named_modules():
            weight_initializer = None
            bias_initializer = None
            if isinstance(module, INITIALIZED_MODULE_TYPES):
                if model_type == 'AE':
                    weight_initializer = uniform_quantile_from_weights(num_quantiles, module.weight)
                elif model_type == 'lstm':
                    # Initialize logits layer by LM initializer
                    weight_initializer = uniform_quantile_from_weights(num_quantiles, module.weight)
                elif isinstance(model, FixupResNet):
                    weight_initializer = fixup_resnet_module_weight_quantile(num_quantiles, name,
                                                                                   module, model.num_layers)
                else:
                    weight_initializer = uniform_quantile_from_weights(num_quantiles, module.weight)

                if module.bias is not None:
                    if model_type in 'AE':
                        bias_initializer = uniform_quantile_from_weights(num_quantiles, module.weight)
                    elif model_type == 'lstm':
                        # Initialize logits layer by LM initializer
                        bias_initializer = uniform_quantile_from_weights(num_quantiles, module.bias)
                    elif isinstance(model, FixupResNet):
                        bias_initializer = None
                    else:
                        raise Exception("Unknown model type")
                else:
                    bias_initializer = None

            elif isinstance(module, INITIALIZED_RECURRENT_MODULE_TYPES):
                num_gates = 4 if isinstance(module, nn.LSTMCell) else 3
                hidden_size = module.hidden_size
                weight_initializer = nn.ModuleDict()
                for weight_name, weights in module._parameters.items():
                    if weight_name == 'bias_hh': continue
                    for gate_id in range(num_gates):
                        gate_name = '_'.join((weight_name, 'gate_{}'.format(gate_id)))

                        quantile_function = kaiming_uniform_quantile_given_fan(num_quantiles, hidden_size)
                        weight_initializer[gate_name] = quantile_function

            elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
                weight_initializer = normal_quantile(num_quantiles, std=1.0)

            key_name = "_".join(name.split("."))
            assert key_name not in self.initializers.keys(), 'model has to have unique layer names'
            if weight_initializer is None and bias_initializer is None:
                if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                    print(name)
            else:
                self.initializers[key_name] = nn.ModuleList([weight_initializer, bias_initializer])
                self.key_name2name[key_name] = name

        self.untrained_initializers = {key: deepcopy(value) for key, value in self.initializers.items()}
