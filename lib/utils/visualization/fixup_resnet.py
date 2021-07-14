import torch
import numpy as np
import matplotlib.pyplot as plt
from lib.utils import moving_average, check_numpy


@torch.no_grad()
def visualize_pdf(maml):
    i = 0
    plt.figure(figsize=[22, 34])
    for name, (weight_maml_init, bias_maml_init) in maml.initializers.items():
        weight_base_init, _ = maml.untrained_initializers[name]
        base_mean = weight_base_init.mean.item()
        base_std = weight_base_init.std.item()
        maml_mean = weight_maml_init.mean.item()
        maml_std = weight_maml_init.std.item()
        
        base_init = torch.distributions.Normal(base_mean, base_std)
        maml_init = torch.distributions.Normal(maml_mean, maml_std)
        i += 1
        plt.subplot(6, 4, i)
        xx = np.linspace(min([base_mean - 3.*base_std, maml_mean - 3.*maml_std]), 
                            max([base_mean + 3.*base_std, maml_mean + 3.*maml_std]), 1000)

        if i == 12:
            yy = base_init.log_prob(torch.tensor(xx)).exp().numpy()
            plt.plot(xx, yy, '--', label='Fixup')
            yy = maml_init.log_prob(torch.tensor(xx)).exp().numpy()
            plt.plot(xx, yy, c='g', label='Fixup + DIMAML')
            leg = plt.legend(loc=4, fontsize=14.5, frameon=False)
            for line in leg.get_lines():
                line.set_linewidth(1.6)
        else:
            yy = base_init.log_prob(torch.tensor(xx)).exp().numpy()
            plt.plot(xx, yy, '--')
            yy = maml_init.log_prob(torch.tensor(xx)).exp().numpy()
            plt.plot(xx, yy, c='g')
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(name + '_weight', fontsize=14)
    plt.show()


@torch.no_grad()
def visualize_quantile_functions(maml):
    plt.figure(figsize=[22, 34])
    i = 0
    for name, (weight_quantile_function, bias_quantile_function) in maml.initializers.items():
        wq_init, bq_init = maml.untrained_initializers[name]
        i += 1
        plt.subplot(6, 4, i)
        xx = torch.linspace(0., 1., 1000).cuda()
        if i == 12:
            yy = wq_init(xx)
            plt.plot(check_numpy(xx), check_numpy(yy), '--', label='Fixup')
            yy = weight_quantile_function(xx)
            plt.plot(check_numpy(xx), check_numpy(yy), c='g', label='Fixup $\\rightarrow$ DIMAML')
            leg = plt.legend(loc=4, fontsize=14, frameon=False)
            for line in leg.get_lines():
                line.set_linewidth(1.6)
        else:
            yy = wq_init(xx)
            plt.plot(check_numpy(xx), check_numpy(yy), '--')
            yy = weight_quantile_function(xx)
            plt.plot(check_numpy(xx), check_numpy(yy), c='g')
        
        plt.xlim([0, 1])
        plt.title(name + '_weight')
    plt.show()


def draw_plots(base_train_loss, base_test_loss, base_test_error,
               maml_train_loss, maml_test_loss, maml_test_error):
    plt.figure(figsize=(20, 6))
    plt.subplot(1,3,1)
    plt.plot(moving_average(base_train_loss, span=10), label='Baseline')
    plt.plot(moving_average(maml_train_loss, span=10), c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Train loss", fontsize=14)
    plt.subplot(1,3,2)
    plt.plot(base_test_loss, label='Baseline')
    plt.plot(maml_test_loss, c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Test loss", fontsize=14)
    plt.subplot(1,3,3)
    plt.plot(base_test_error, label='Baseline')
    plt.plot(maml_test_error, c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Test classification error", fontsize=14)