"""
Utilities required for backpropagating through gradient descent steps, inspired by:
Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks https://arxiv.org/abs/1703.03400
"""
from collections import namedtuple
from warnings import warn

import torch
from torch import nn as nn

import math
from itertools import chain
from .utils.general_utils import straight_through_grad, copy_and_replace, NONE_TENSOR, is_none_tensor
from .models.fixup_resnet import FixupResNet


def make_inner_optimizer(optimizer_type, **kwargs):
    if optimizer_type == 'sgd':
        optimizer = IngraphGradientDescent(**kwargs)
    elif optimizer_type == 'momentum':
        optimizer = IngraphMomentum(**kwargs)
    elif optimizer_type == 'adam':
        optimizer = IngraphAdam(**kwargs)
    else: 
        raise NotImplemetedError("This optimizer is not implemeted")
    return optimizer


def make_eval_inner_optimizer(maml, model, optimizer_type, **kwargs):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(maml.get_parameters(model), **kwargs)
    elif optimizer_type == 'momentum':
        optimizer = torch.optim.SGD(maml.get_parameters(model), **kwargs)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(maml.get_parameters(model), **kwargs)
    else: 
        raise NotImplemetedError("{} optimizer is not implemeted".format(optimizer_type))
    return optimizer


def get_updated_model(model: nn.Module, loss=None, gradients=None, parameters=None,
                      detach=False, learning_rate=1.0, allow_unused=False, **kwargs):
    """
    Creates a copy of model whose parameters are updated with one-step gradient descent w.r.t. loss
    The copy will propagate gradients into the original model
    :param model: original model
    :param loss: scalar objective to backprop from; provide either this or gradients
    :param gradients: a list or tuple of gradients (updates) for each parameter; provide either this or loss
    :param parameters: list/tuple of parameters to update, defaults to model.parameters()
    :param detach: if True, the resulting model will not propagate gradients to the original model
    :param learning_rate: scales gradients by this value before updating
    :param allow_unused: by default, raise an error if one or more parameters receive None gradients
        Otherwise (allow_unused=True) simply do not update these parameters
    """
    assert (loss is None) != (gradients is None)
    parameters = list(model.parameters() if parameters is None else parameters)

    if gradients is None:
        assert torch.is_grad_enabled()
        gradients = torch.autograd.grad(
            loss, parameters, create_graph=not detach, only_inputs=True, allow_unused=allow_unused, **kwargs)

    assert isinstance(gradients, (list, tuple)) and len(gradients) == len(parameters)

    gradients = list(gradients)
    updates = []
    for weight, grad in zip(parameters, gradients):
        if grad is not None:
            # If weight corresponds to scale or bias in Fixup, change lr
            if isinstance(model, FixupResNet) and weight.shape == torch.Size([1, 1]):
                update = weight - 0.1 * learning_rate * grad
            else:
                update = weight - learning_rate * grad
            if detach:
                update = update.detach().requires_grad_(weight.requires_grad)
            updates.append(update)

    updates = dict(zip(parameters, updates))

    do_not_copy = [tensor for tensor in chain(model.parameters(), model.buffers())
                   if tensor not in updates]

    return copy_and_replace(model, updates, do_not_copy)


class IngraphGradientDescent(nn.Module):
    """ Optimizer that updates model out-of-place and returns a copy with changed parameters """
    OptimizerState = namedtuple("OptimizerState", [])

    def __init__(self, lr=0.1):
        super().__init__()
        self.learning_rate = lr

    def get_initial_state(self, module, *, parameters: list, **kwargs):
        """ Return initial optimizer state: momenta, rms, etc. State must be a collection of torch tensors! """
        return self.OptimizerState()

    def step(self, state: OptimizerState, module: nn.Module, loss, parameters=None, **kwargs):
        """
        Return an updated copy of model after one iteration of gradient descent
        :param state: optimizer state (as in self.get_initial_state)
        :param module: module to be updated
        :param loss: torch scalar that is differentiable w.r.t. model parameters
        :parameters: parameters of :module: that will be edited by updates (default = module.parameters())
        :param kwargs: extra parameters passed to get_updated_model
        :returns: new_state, updated_self
            new_state: self.OptimizerState - optimizer state after performing sgd step
            updated_self: updated(out-of-place) version of self
        """
        updated_model = get_updated_model(module, loss=loss, learning_rate=self.learning_rate,
                                          parameters=list(parameters or module.parameters()), **kwargs)
        return state, updated_model

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)


class IngraphMomentum(IngraphGradientDescent):
    OptimizerState = namedtuple(
        "OptimizerState", ["grad_momenta", "learning_rate", "momentum", "weight_decay"])

    def __init__(self, lr=0.1, momentum=0.9, weight_decay=0, 
                 nesterov=False, force_trainable_params=False):
        """
        Ingraph optimizer that performs SGD updates with momentum
        :param lr: learning rate
        :param momentum: momentum coefficient, the update direction is (1 - momentum) * prev_update  + update,
            default = no momentum
        :param weight_decay: weight decay (L2 penalty)
        :param nesterov: enables Nesterov momentum
        :param force_trainable_params: if True, treats all optimizer parameters that are not None as learnable
            parameters that are trained alongside other non-edited layers
        """
        nn.Module.__init__(self)
        weight_decay = weight_decay if weight_decay > 0. else None
        self.hparams = dict(learning_rate=lr, momentum=momentum, weight_decay=weight_decay)

        if force_trainable_params:
            for key in self.hparams:
                if self.hparams[key] is None:
                    continue
                elif isinstance(self.hparams[key], nn.Parameter):
                    continue
                elif isinstance(self.hparams[key], torch.Tensor) and self.hparams[key].requires_grad:
                    continue
                self.hparams[key] = nn.Parameter(torch.as_tensor(self.hparams[key]))

        for key in self.hparams:
            if isinstance(self.hparams[key], nn.Parameter):
                self.register_parameter(key, self.hparams[key])

        self.nesterov = nesterov

    def get_initial_state(self, module: nn.Module, *, parameters: list, **overrides):
        """
        Create initial state and make sure all parameters are in a valid range.
        State:
        * must be a (nested) collection of torch tensors : lists/tuples/dicts/namedtuples of tensors or lists/... of them
        * the structure (i.e. lengths) of this collection should NOT change between iterations.
        * the optimizer state at the input of :step: method forces requires_grad=True to all tensors.

        :param module: module to be updated
        :param parameters: list of trainable parameters
        :param overrides: send key-value optimizer params with same names as at init to override them
        :return: self.OptimizerState
        """
        for key in overrides:
            assert key in self.hparams, "unknown optimizer parameter {}".format(key)
        hparams = dict(self.hparams, **overrides)

        learning_rate = hparams['learning_rate']
        learning_rate = straight_through_grad(torch.clamp_min, min=0.0)(torch.as_tensor(learning_rate))

        momentum = hparams.get('momentum')
        if momentum is not None:
            momentum = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(momentum))
        else:
            momentum = NONE_TENSOR

        weight_decay = hparams.get('weight_decay')
        if weight_decay is not None:
            weight_decay = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(weight_decay))
        else:
            weight_decay = NONE_TENSOR

        if isinstance(momentum, torch.Tensor) and momentum.requires_grad:
            warn("The derivative of updated params w.r.t. momentum is proportional to momentum^{n_steps - 1}, "
                 "optimizing it with gradient descent may suffer from poor numerical stability.")

        dummy_grad_momenta = [NONE_TENSOR for _ in parameters]
        return self.OptimizerState(dummy_grad_momenta, learning_rate, momentum, weight_decay)

    def step(self, state: OptimizerState, module: nn.Module, loss, parameters=None, detach=False, **kwargs):
        """
        :param state: optimizer state (as in self.get_initial_state)
        :param module: module to be edited
        :param loss: torch scalar that is differentiable w.r.t. model parameters
        :param parameters: if model
        :param kwargs: extra parameters passed to get_updated_model
        :returns: new_state, updated_self
            new_state: self.OptimizerState - optimizer state after performing sgd step
            updated_self: updated copy of module
        """
        grad_momenta, learning_rate, momentum, weight_decay = state
        learning_rate, momentum, weight_decay = [tensor.to(loss.device)
                                                 for tensor in (learning_rate, momentum, weight_decay)]
        parameters = list(parameters or module.parameters())
        gradients = list(torch.autograd.grad(loss, parameters, create_graph=not detach,
                                             only_inputs=True, allow_unused=False))

        for i, grad in enumerate(gradients):
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"Nan or Inf gradients")
                # raise Exception(f"Nan or Inf gradients {loss}")
            gradients[i] = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
            gradients[i] = torch.where(torch.isinf(grad), torch.zeros_like(grad), grad)

        if not is_none_tensor(weight_decay) and float(weight_decay) != 0:
            for i in range(len(gradients)):
                gradients[i] = gradients[i] + weight_decay * parameters[i]

        updates = gradients  # updates are the scaled/accumulated/tuned gradients

        if not is_none_tensor(momentum) and float(momentum) != 0:
            # momentum: accumulate gradients with moving average-like procedure
            if is_none_tensor(grad_momenta[0]):
                grad_momenta = list(gradients)
            else:
                for i in range(len(grad_momenta)):
                    grad_momenta[i] = grad_momenta[i] * momentum + gradients[i]

            if self.nesterov:
                for i in range(len(grad_momenta)):
                    updates[i] = updates[i] + momentum * grad_momenta[i]
            else:
                updates = grad_momenta

        # finally, perform sgd update
        updated_module = get_updated_model(module, loss=None, gradients=updates, parameters=parameters,
                                           learning_rate=learning_rate, **kwargs)
        new_state = self.OptimizerState(grad_momenta, learning_rate, momentum, weight_decay)
        return new_state, updated_module

    def extra_repr(self):
        return repr(self.hparams)


class IngraphAdam(IngraphGradientDescent):
    OptimizerState = namedtuple("OptimizerState", ["step", "grad_momenta", "ewma_grad_norms_sq", "learning_rate",
                                                   "beta1", "beta2", "epsilon", "weight_decay"])

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0., force_trainable_params=False):
        """
        Ingraph optimizer that performs Adam updates
        :param lr: learning rate
        :param betas: coefficients used for computing running averages of gradient and its square
        :param eps: term added to the denominator to improve numerical stability 
        :param weight_decay: weight decay (L2 penalty)
        :param force_trainable_params: if True, treats all optimizer parameters that are not None as learnable
            parameters that are trained alongside other non-edited layers
        """
        nn.Module.__init__(self)
        weight_decay = weight_decay if weight_decay > 0. else None
        self.hparams = dict(learning_rate=lr, beta1=betas[0], beta2=betas[1],
                            epsilon=eps, weight_decay=weight_decay)

        if force_trainable_params:
            for key in self.hparams:
                if self.hparams[key] is None:
                    continue
                elif isinstance(self.hparams[key], nn.Parameter):
                    continue
                elif isinstance(self.hparams[key], torch.Tensor) and self.hparams[key].requires_grad:
                    continue
                self.hparams[key] = nn.Parameter(torch.as_tensor(self.hparams[key]))

        for key in self.hparams:
            if isinstance(self.hparams[key], nn.Parameter):
                self.register_parameter(key, self.hparams[key])

    def get_initial_state(self, module: nn.Module, *, parameters: list, **overrides):
        """
        Create initial state and make sure all parameters are in a valid range.
        State:
        * must be a (nested) collection of torch tensors : lists/tuples/dicts/namedtuples of tensors or lists/... of them
        * the structure (i.e. lengths) of this collection should NOT change between iterations.
        * the optimizer state at the input of :step: method forces requires_grad=True to all tensors.

        :param module: module to be updated
        :param parameters: list of trainable parameters
        :param overrides: send key-value optimizer params with same names as at init to override them
        :return: self.OptimizerState
        """
        for key in overrides:
            assert key in self.hparams, "unknown optimizer parameter {}".format(key)
        hparams = dict(self.hparams, **overrides)

        learning_rate = hparams['learning_rate']
        learning_rate = straight_through_grad(torch.clamp_min, min=0.0)(torch.as_tensor(learning_rate))

        weight_decay = hparams.get('weight_decay')
        if weight_decay is not None:
            weight_decay = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(weight_decay))
        else:
            weight_decay = NONE_TENSOR

        beta1 = hparams.get('beta1')
        if beta1 is not None:
            beta1 = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(beta1))
        else:
            beta1 = NONE_TENSOR

        if isinstance(beta1, torch.Tensor) and beta1.requires_grad:
            warn("The derivative of updated params w.r.t. momentum is proportional to momentum^{n_steps - 1}, "
                 "optimizing it with gradient descent may suffer from poor numerical stability.")

        beta2 = hparams.get('beta2')
        if beta2 is not None:
            beta2 = straight_through_grad(torch.clamp, min=0.0, max=1.0)(torch.as_tensor(beta2))

            if hparams['epsilon'] is None:
                hparams['epsilon'] = 1e-8
            epsilon = torch.as_tensor(hparams['epsilon'])
            epsilon = straight_through_grad(torch.clamp_min, min=1e-9)(torch.as_tensor(epsilon))
        else:
            beta2 = NONE_TENSOR
            epsilon = NONE_TENSOR

        dummy_grad_momenta = [torch.zeros_like(params, requires_grad=False) for params in parameters]
        dummy_ewma = [torch.zeros_like(params, requires_grad=False) for params in parameters]
        step = torch.zeros(1, requires_grad=False)
        return self.OptimizerState(step, dummy_grad_momenta, dummy_ewma,
                                   learning_rate, beta1, beta2, epsilon, weight_decay)

    def step(self, state: OptimizerState, module: nn.Module, loss, parameters=None, detach=False, **kwargs):
        """
        :param state: optimizer state (as in self.get_initial_state)
        :param module: module to be edited
        :param loss: torch scalar that is differentiable w.r.t. model parameters
        :param parameters: if model
        :param kwargs: extra parameters passed to get_updated_model
        :returns: new_state, updated_self
            new_state: self.OptimizerState - optimizer state after performing sgd step
            updated_self: updated copy of module
        """
        step, grad_momenta, ewma_grad_norms_sq, learning_rate, beta1, beta2, epsilon, weight_decay = state
        step, learning_rate, beta1, beta2, epsilon, weight_decay = [
            tensor.to(loss.device) for tensor in (step, learning_rate, beta1, beta2, epsilon, weight_decay)]
        parameters = list(parameters or module.parameters())
        gradients = list(torch.autograd.grad(loss, parameters, create_graph=not detach, only_inputs=True, allow_unused=False))

        if not is_none_tensor(weight_decay) and float(weight_decay) > 0:
            for i in range(len(gradients)):
                gradients[i] = gradients[i] + weight_decay * parameters[i]

        updates = gradients  # updates are the scaled/accumulated/tuned gradients

        step = step + 1.
        assert step > 0, "'step' must be positive integer"
        if not is_none_tensor(beta1) and beta1 > 0:
            # beta1: accumulate gradients with moving average-like procedure
            bias_correction1 = 1. - beta1 ** step
            assert not is_none_tensor(grad_momenta[0])
            for i in range(len(grad_momenta)):
                grad_momenta[i] = grad_momenta[i] * beta1 + gradients[i] * (1. - beta1)
        else:
            bias_correction1 = 1.
            grad_momenta = gradients

        if not is_none_tensor(beta2):
            bias_correction2 = 1. - beta2 ** step
            assert not is_none_tensor(ewma_grad_norms_sq[0])
            for i in range(len(ewma_grad_norms_sq)):
                ewma_grad_norms_sq[i] = beta2 * ewma_grad_norms_sq[i] + (1. - beta2) * gradients[i] ** 2
                # scale updates by 1 / sqrt(moving_average_norm_squared) + epsilon
                ewma_grad_norms_sq[i][ewma_grad_norms_sq[i] == 0] = 1e-12  # To avoid Nan for meta gradients
                sqrt_exp_avg_sq = torch.sqrt(ewma_grad_norms_sq[i]) / math.sqrt(bias_correction2)
                updates[i] = grad_momenta[i] / (sqrt_exp_avg_sq + epsilon)

        # finally, perform sgd update
        updated_module = get_updated_model(module, loss=None, gradients=updates, parameters=parameters,
                                           learning_rate=learning_rate / bias_correction1, **kwargs)
        new_state = self.OptimizerState(step, grad_momenta, ewma_grad_norms_sq, learning_rate,
                                        beta1, beta2, epsilon, weight_decay)
        return new_state, updated_module

    def extra_repr(self):
        return repr(self.hparams)