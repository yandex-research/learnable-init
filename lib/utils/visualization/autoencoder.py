import torch
import numpy as np
import matplotlib.pyplot as plt
from lib.utils import moving_average, check_numpy


@torch.no_grad()
def visualize_quantile_functions(maml):
    plt.figure(figsize=[20, 32])
    i = 0
    for name, (weight_quantile_function, bias_quantile_function) in maml.initializers.items():
        wq_init, bq_init = maml.untrained_initializers[name]
        i += 1
        plt.subplot(6, 4, i)
        xx = torch.linspace(0., 1., 1000).cuda()
        yy = wq_init(xx)
        plt.plot(check_numpy(xx), check_numpy(yy), '--')
        yy = weight_quantile_function(xx)
        plt.plot(check_numpy(xx), check_numpy(yy), c='g')
        plt.xlim([0, 1])
        plt.title(name + '_weight', fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        if i in [9, 10, 11, 12]:
            plt.xlabel("z$\sim$U(0,1)", fontsize=14)
                
        if bias_quantile_function is not None:
            i += 1
            plt.subplot(6, 4, i)
            if i == 12:
                plt.plot(check_numpy(xx), check_numpy(yy), '--', label='Kaiming')
                yy = bias_quantile_function(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), c='g', label='DIMAML')
                leg = plt.legend(loc=4, fontsize=15, frameon=False)
                for line in leg.get_lines():
                    line.set_linewidth(1.6)
            else:
                yy = bq_init(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), '--',)
                yy = bias_quantile_function(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), c='g')
            plt.xlim([0, 1])
            plt.title(name + '_bias', fontsize=14)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
        
            if i in [9, 10, 11, 12]:
                plt.xlabel("z$\sim$U(0,1)", fontsize=14)
    plt.show()
        

@torch.no_grad()
def visualize_pdf(maml):
    plt.figure(figsize=[22, 34])
    i = 0
    for name, (weight_maml_init, bias_maml_init) in maml.initializers.items():
        weight_base_init, bias_base_init = maml.untrained_initializers[name]
        
        weight_base_mean = weight_base_init.mean.item()
        weight_base_std = weight_base_init.std.item()
        weight_base_init = torch.distributions.Normal(weight_base_mean, abs(weight_base_std))
        
        weight_maml_mean = weight_maml_init.mean.item()
        weight_maml_std = weight_maml_init.std.item()
        weight_maml_init = torch.distributions.Normal(weight_maml_mean, abs(weight_maml_std))
        
        xx = np.linspace(min([weight_base_mean - 3.*weight_base_std, 
                                weight_maml_mean - 3.*weight_maml_std]), 
                            max([weight_base_mean + 3.*weight_base_std, 
                                weight_maml_mean + 3.*weight_maml_std]), 1000)
        i += 1
        plt.subplot(6, 4, i)
        yy = weight_base_init.log_prob(torch.tensor(xx)).exp().numpy()
        plt.plot(xx, yy, '--')
        yy = weight_maml_init.log_prob(torch.tensor(xx)).exp().numpy()
        plt.plot(xx, yy, c='g')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(name + '_weight', fontsize=14)
                
        if bias_maml_init is not None:
            bias_base_mean = bias_base_init.mean.item()
            bias_base_std = bias_base_init.std.item()
            bias_base_init = torch.distributions.Normal(bias_base_mean, bias_base_std)

            bias_maml_mean = bias_maml_init.mean.item()
            bias_maml_std = bias_maml_init.std.item()
            bias_maml_init = torch.distributions.Normal(bias_maml_mean, bias_maml_std)
        
            i += 1
            plt.subplot(6, 4, i)
            if i == 12:
                yy = bias_base_init.log_prob(torch.tensor(xx)).exp().numpy()
                plt.plot(xx, yy, '--', label='Kaiming')
                yy = bias_maml_init.log_prob(torch.tensor(xx)).exp().numpy()
                plt.plot(xx, yy, c='g', label='DIMAML')
                leg = plt.legend(loc=4, fontsize=15, frameon=False)
                for line in leg.get_lines():
                    line.set_linewidth(1.6)
            else:
                yy = bias_base_init.log_prob(torch.tensor(xx)).exp().numpy()
                plt.plot(xx, yy, '--')
                yy = bias_maml_init.log_prob(torch.tensor(xx)).exp().numpy()
                plt.plot(xx, yy, c='g')
                
            plt.title(name + '_bias', fontsize=14)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
    plt.show()


def draw_plots(base_train_loss, base_test_loss, maml_train_loss, maml_test_loss):
    plt.figure(figsize=(16, 6))
    plt.subplot(1,2,1)
    plt.plot(moving_average(base_train_loss, span=10), label='Baseline')
    plt.plot(moving_average(maml_train_loss, span=10), c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Train loss", fontsize=14)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    
    plt.subplot(1,2,2)
    plt.plot(base_test_loss, label='Baseline')
    plt.plot(maml_test_loss, c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Test loss", fontsize=14)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("MSE", fontsize=14)