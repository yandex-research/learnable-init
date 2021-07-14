import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from lib.utils import moving_average, check_numpy


@torch.no_grad()
def visualize_pdf(maml):
    indices = [-1, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 6, 
                17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30]
    plt.figure(figsize=[22, 26])
    i = 0
    for name, (weight_maml_init, bias_maml_init) in maml.initializers.items():
        weight_base_init, bias_base_init = maml.untrained_initializers[name]
        if not isinstance(weight_maml_init, nn.ModuleDict):
            weight_base_mean = weight_base_init.mean.item()
            weight_base_std = weight_base_init.std.item()
            weight_base_init = torch.distributions.Normal(weight_base_mean, weight_base_std)

            weight_maml_mean = weight_maml_init.mean.item()
            weight_maml_std = weight_maml_init.std.item()
            weight_maml_init = torch.distributions.Normal(weight_maml_mean, weight_maml_std)

            xx = np.linspace(min([weight_base_mean - 3.*weight_base_std, 
                                    weight_maml_mean - 3.*weight_maml_std]), 
                                max([weight_base_mean + 3.*weight_base_std, 
                                    weight_maml_mean + 3.*weight_maml_std]), 1000)
            i += 1
            plt.subplot(6, 5, indices[i])
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
                plt.subplot(6, 5, indices[i])
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
        else:
            for weight_name, maml_init in weight_maml_init.items():
                base_init = weight_base_init[weight_name]
                
                weight_name_split = weight_name.split('_')
                if weight_name_split[-1] == '0':
                    gate_name = 'input_gate'
                elif weight_name_split[-1] == '1':
                    gate_name = 'forget_gate'
                elif weight_name_split[-1] == '2':
                    gate_name = 'update'
                elif weight_name_split[-1] == '3':
                    gate_name = 'output_gate'
                    
                weight_base_mean = base_init.mean.item()
                weight_base_std = base_init.std.item()
                base_init = torch.distributions.Normal(weight_base_mean, weight_base_std)

                weight_maml_mean = maml_init.mean.item()
                weight_maml_std = maml_init.std.item()
                maml_init = torch.distributions.Normal(weight_maml_mean, weight_maml_std)

                xx = np.linspace(min([weight_base_mean - 3.*weight_base_std, 
                                    weight_maml_mean - 3.*weight_maml_std]), 
                            max([weight_base_mean + 3.*weight_base_std, 
                                weight_maml_mean + 3.*weight_maml_std]), 1000)
                i += 1
                plt.subplot(6, 5, indices[i])
                yy = base_init.log_prob(torch.tensor(xx)).exp().numpy()
                plt.plot(xx, yy, '--')
                yy = maml_init.log_prob(torch.tensor(xx)).exp().numpy()
                plt.plot(xx, yy, c='g')
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

                if weight_name_split[0] == 'weight':
                    weight_name = '_'.join(['lstm', gate_name] + weight_name_split[:2])
                else:
                    weight_name = '_'.join(['lstm', gate_name, weight_name_split[0]])
                plt.title(weight_name, fontsize=14)               
    plt.show()


@torch.no_grad()
def visualize_quantile_functions(maml):
    indices = [-1, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 6, 
                17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30]
    plt.figure(figsize=[22, 26])
    
    i = 0
    for name, (weight_quantile_function, bias_quantile_function) in maml.initializers.items():
        if not isinstance(weight_quantile_function, nn.ModuleDict):
            wq_init, bq_init = maml.untrained_initializers[name]
            i += 1
            plt.subplot(6, 5, indices[i])
            xx = torch.linspace(0., 1., 1000).cuda()
            
            if name == 'logits':
                plt.title('Logit weights', fontsize=13)
                yy = wq_init(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), '--')
                yy = weight_quantile_function(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), c='g')
            else:
                plt.title('Embeddings', fontsize=13)
                yy = wq_init(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), '--', label='N(0,1)')
                yy = weight_quantile_function(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), c='g', label='DIMAML')
                
                leg = plt.legend(loc=4, fontsize=15, frameon=False)
                for line in leg.get_lines():
                    line.set_linewidth(1.2) 
                        
            if bias_quantile_function is not None:
                i += 1
                plt.subplot(6, 5, indices[i])
                yy = bq_init(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), '--')
                yy = bias_quantile_function(xx)
                plt.plot(check_numpy(xx), check_numpy(yy), c='g')
                plt.title('Logits bias', fontsize=13)   
        else:
            wq_init, bq_init = maml.untrained_initializers[name]
            for weight_name, quantile_function in weight_quantile_function.items():
                i += 1
                plt.subplot(6, 5, indices[i])
                
                weight_name_split = weight_name.split('_')
                if weight_name_split[-1] == '0':
                    gate_name = 'input_gate'
                elif weight_name_split[-1] == '1':
                    gate_name = 'forget_gate'
                elif weight_name_split[-1] == '2':
                    gate_name = 'update'
                elif weight_name_split[-1] == '3':
                    gate_name = 'output_gate'
                
                if weight_name_split[-1] in ['0', '1', '2'] or weight_name_split[0] == 'weight':
                    xx = torch.linspace(0., 1., 1000).cuda()
                    yy = wq_init[weight_name](xx)
                    plt.plot(check_numpy(xx), check_numpy(yy), '--')
                    yy = quantile_function(xx)
                    plt.plot(check_numpy(xx), check_numpy(yy), c='g')
                elif weight_name_split[0] == 'bias':
                    xx = torch.linspace(0., 1., 1000).cuda()
                    yy = wq_init[weight_name](xx)
                    if 'lstm2' in name:
                        plt.plot(check_numpy(xx), check_numpy(yy), '--', label='Kaiming')
                        yy = quantile_function(xx)
                        plt.plot(check_numpy(xx), check_numpy(yy), c='g', label='DIMAML')
                        leg = plt.legend(loc=4, fontsize=15, frameon=False)
                        for line in leg.get_lines():
                            line.set_linewidth(1.2) 
                    else:
                        plt.plot(check_numpy(xx), check_numpy(yy), '--')
                        yy = quantile_function(xx)
                        plt.plot(check_numpy(xx), check_numpy(yy), c='g')
                                
                if weight_name_split[0] == 'weight':
                    weight_name = '_'.join(['lstm', gate_name] + weight_name_split[:2])
                else:
                    weight_name = '_'.join(['lstm', gate_name, weight_name_split[0]])
                plt.title(weight_name, fontsize=13)
    plt.show()


def draw_plots(base_train_loss, base_test_loss, maml_train_loss, maml_test_loss):
    plt.figure(figsize=(20,6))
    plt.subplot(1,2,1)
    plt.plot(moving_average(base_train_loss, span=10), label='Baseline')
    plt.plot(moving_average(maml_train_loss, span=10), c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Train loss", fontsize=14)
    plt.subplot(1,2,2)
    plt.plot(base_test_loss, label='Baseline')
    plt.plot(maml_test_loss, c='g', label='DIMAML')
    plt.legend(fontsize=14)
    plt.title("Test loss", fontsize=14)
    plt.show()