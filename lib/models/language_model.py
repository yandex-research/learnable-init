import torch, torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class LanguageModel(nn.Module):
    """ 
        While cuDNN LSTM does not allow for second backpropagation,
        we use a less efficient LSTM implemetation for DIMAML.
    """
    def __init__(self, voc_size, emb_size, hid_size, drop_prob=0.1):
        super().__init__()
        self.emb_vectors = nn.Embedding(voc_size, emb_size)
        self.lstm1 = nn.LSTMCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, voc_size)
        self.lstm2 = nn.LSTMCell(hid_size, hid_size)
        self.dropout = nn.Dropout(drop_prob)

        self.init_weights()

    def init_weights(self):
        self.logits.bias.data.zero_()

    def forward(self, inputs):
        # indices shape: [batch_size, seq_length]
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[-1]
        embed = self.emb_vectors(inputs)

        h1,c1 = [torch.zeros(batch_size,self.lstm1.hidden_size, device=embed.device),
                 torch.zeros(batch_size,self.lstm1.hidden_size, device=embed.device)]
        h2,c2 = [torch.zeros(batch_size,self.lstm2.hidden_size, device=embed.device),
                 torch.zeros(batch_size,self.lstm2.hidden_size, device=embed.device)]

        hid_seq = []
        for step in range(seq_length):
            h1,c1 = self.lstm1(embed[:, step], (h1, c1))
            h2,c2 = self.lstm2(self.dropout(h1), (h2, c2))
            hid_seq.append(h2.unsqueeze(1))

        hid_seq = self.dropout(torch.cat(hid_seq, dim=1))
        outputs = self.logits(hid_seq)         # (batch_size, max_len, voc_size)
        return F.log_softmax(outputs, dim=-1)  # [batch_size, seq_length, voc_size]


class CudnnLanguageModel(nn.Module):
    """ Uses cuDNN LSTM for more efficient initializer evaluation """
    def __init__(self, voc_size, emb_size, hid_size, drop_prob=0.1, num_layers=2):
        super().__init__()
        self.emb_vectors = nn.Embedding(voc_size, emb_size)
        self.logits = nn.Linear(hid_size, voc_size)

        self.lstm = nn.LSTM(emb_size, hid_size, num_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        self.logits = nn.Linear(hid_size, voc_size)
        self.num_layers = num_layers

    def forward(self, inputs, initial_state=None):
        # indices shape: [batch_size, seq_length]
        embs = self.emb_vectors(inputs)

        if initial_state is None:
            hidden = (torch.zeros(self.num_layers, inputs.shape[0],
                                  self.lstm.hidden_size, device=embs.device),
                      torch.zeros(self.num_layers, inputs.shape[0],
                                  self.lstm.hidden_size, device=embs.device))

        r_output, hidden = self.lstm(embs, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.lstm.hidden_size)
        logits = self.logits(out)
        return F.log_softmax(logits, dim=-1)  # [batch_size, seq_length, voc_size]


def convert_to_cudnn_lstm(model):
    voc_size, hid_size = model.logits.weight.shape
    emb_size = model.emb_vectors.weight.shape[1]
    device = model.emb_vectors.weight.device
    cudnn_model = CudnnLanguageModel(voc_size, emb_size, hid_size)

    cudnn_model.emb_vectors = deepcopy(model.emb_vectors)
    cudnn_model.logits = deepcopy(model.logits)

    for name, weights in model.lstm1.named_parameters():
        cudnn_model.lstm._parameters[name + '_l0'] = deepcopy(weights)
    cudnn_model.lstm._parameters['bias_hh_l0'].requires_grad = False
    cudnn_model.lstm._parameters['bias_hh_l0'].zero_()

    for name, weights in model.lstm2.named_parameters():
        cudnn_model.lstm._parameters[name + '_l1'] = deepcopy(weights)
    cudnn_model.lstm._parameters['bias_hh_l1'].requires_grad = False
    cudnn_model.lstm._parameters['bias_hh_l1'].zero_()

    cudnn_model = cudnn_model.to(device)
    cudnn_model.lstm.flatten_parameters()
    return cudnn_model
