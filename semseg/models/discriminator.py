import torch
import torch.nn as nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
