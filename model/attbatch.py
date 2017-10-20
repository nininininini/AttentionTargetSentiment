# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Att(nn.Module):
    def __init__(self, input_size, output_size, cuda):
        super(Att, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size * 2, output_size, bias=True)
        nn.init.xavier_uniform(self.linear.weight)

        if cuda:
            self.u = Variable(torch.randn(output_size, 1).cuda(), requires_grad=True)
        else:
            self.u = Variable(torch.randn(output_size, 1), requires_grad=True)
        nn.init.xavier_uniform(self.u)

    def forward(self, h, attention):
        m_combine = torch.cat([h, torch.cat([attention for _ in range(h.size(0))], 0)], 1)

        m_combine = F.tanh(self.linear(m_combine))

        beta = torch.mm(m_combine, self.u)
        beta = torch.t(beta)

        alfa = F.softmax(beta)

        output = torch.mm(alfa, h)

        return output