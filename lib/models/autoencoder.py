import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, latent_dim, planes=32):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(3, planes, 5, stride=2, padding=1, bias=False)  # [batch, 32, 16, 16]
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, 2 * planes, 5, stride=2, padding=1, bias=False)  # [batch, 64, 8, 8]
        self.bn2 = nn.BatchNorm2d(2 * planes)

        self.conv3 = nn.Conv2d(2 * planes, 4 * planes, 5, stride=2, padding=1, bias=False)  # [batch, 128, 7, 7]
        self.bn3 = nn.BatchNorm2d(4 * planes)

        self.conv4 = nn.Conv2d(4 * planes, 8 * planes, 5, stride=2, padding=1, bias=False)  # [batch, 256, 3, 3]
        self.bn4 = nn.BatchNorm2d(8 * planes)

        self.encoder_fc = nn.Linear(8 * planes * 3 * 3, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 4 * planes * 3 * 3, bias=False)

        self.bn5 = nn.BatchNorm2d(4 * planes)

        self.convT1 = nn.ConvTranspose2d(4 * planes, 2 * planes, 5, stride=2, padding=1,
                                         bias=False)  # [batch, 128, 8, 8]
        self.bn6 = nn.BatchNorm2d(2 * planes)

        self.convT2 = nn.ConvTranspose2d(2 * planes, planes, 5, stride=2, padding=1, bias=False)  # [batch, 64, 16, 16]
        self.bn7 = nn.BatchNorm2d(planes)

        self.convT3 = nn.ConvTranspose2d(planes, planes // 2, 5, stride=2, padding=1, bias=False)  # [batch, 32, 16, 16]
        self.bn8 = nn.BatchNorm2d(planes // 2)

        self.convT4 = nn.ConvTranspose2d(planes // 2, 3, 4, stride=2, padding=0, bias=True)  # [batch, 3, 32, 32]

    def forward(self, x):
        h1 = self.activation(self.bn1(self.conv1(x)))
        h2 = self.activation(self.bn2(self.conv2(h1)))
        h3 = self.activation(self.bn3(self.conv3(h2)))
        h4 = self.activation(self.bn4(self.conv4(h3)))

        z = self.encoder_fc(h4.view(-1, 256 * 3 * 3))

        h5 = self.decoder_fc(z).view(-1, 128, 3, 3)
        h5 = self.activation(self.bn5(h5))
        h6 = self.activation(self.bn6(self.convT1(h5)))
        h7 = self.activation(self.bn7(self.convT2(h6)))
        h8 = self.activation(self.bn8(self.convT3(h7)))
        return self.convT4(h8)