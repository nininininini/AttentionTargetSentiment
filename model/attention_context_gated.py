# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .att import Att
from torch.autograd import Variable
from torch.nn import Parameter


class AttentionContextGated(nn.Module):
    def __init__(self, args, m_embedding):
        super(AttentionContextGated, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embedding.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.bilstm = nn.LSTM(args.embed_dim, args.hidden_size,
                              dropout=args.dropout_rnn,
                              batch_first=True,
                              bidirectional=False)

        self.attention = Att(args.hidden_size, args.attention_size, args.cuda)
        self.attention_l = Att(args.hidden_size, args.attention_size, args.cuda)
        self.attention_r = Att(args.hidden_size, args.attention_size, args.cuda)

        if args.cuda:
            self.w1 = Parameter(torch.randn(args.hidden_size, args.hidden_size).cuda(), requires_grad=True)
            self.w2 = Parameter(torch.randn(args.hidden_size, args.hidden_size).cuda(), requires_grad=True)
            self.w3 = Parameter(torch.randn(args.hidden_size, args.hidden_size).cuda(), requires_grad=True)
            self.u1 = Parameter(torch.randn(args.hidden_size, args.hidden_size).cuda(), requires_grad=True)
            self.u2 = Parameter(torch.randn(args.hidden_size, args.hidden_size).cuda(), requires_grad=True)
            self.u3 = Parameter(torch.randn(args.hidden_size, args.hidden_size).cuda(), requires_grad=True)
            self.b1 = Parameter(torch.randn(args.hidden_size, 1).cuda(), requires_grad=True)
            self.b2 = Parameter(torch.randn(args.hidden_size, 1).cuda(), requires_grad=True)
            self.b3 = Parameter(torch.randn(args.hidden_size, 1).cuda(), requires_grad=True)
        else:
            self.w1 = Parameter(torch.randn(args.hidden_size, args.hidden_size), requires_grad=True)
            self.w2 = Parameter(torch.randn(args.hidden_size, args.hidden_size), requires_grad=True)
            self.w3 = Parameter(torch.randn(args.hidden_size, args.hidden_size), requires_grad=True)
            self.u1 = Parameter(torch.randn(args.hidden_size, args.hidden_size), requires_grad=True)
            self.u2 = Parameter(torch.randn(args.hidden_size, args.hidden_size), requires_grad=True)
            self.u3 = Parameter(torch.randn(args.hidden_size, args.hidden_size), requires_grad=True)
            self.b1 = Parameter(torch.randn(args.hidden_size, 1), requires_grad=True)
            self.b2 = Parameter(torch.randn(args.hidden_size, 1), requires_grad=True)
            self.b3 = Parameter(torch.randn(args.hidden_size, 1), requires_grad=True)

        nn.init.xavier_uniform(self.w1)
        nn.init.xavier_uniform(self.w2)
        nn.init.xavier_uniform(self.w3)
        nn.init.xavier_uniform(self.u1)
        nn.init.xavier_uniform(self.u2)
        nn.init.xavier_uniform(self.u3)
        nn.init.xavier_uniform(self.b1)
        nn.init.xavier_uniform(self.b2)
        nn.init.xavier_uniform(self.b3)

        self.linear_2 = nn.Linear(args.hidden_size, args.label_num, bias=True)
        nn.init.xavier_uniform(self.linear_2.weight)

    def forward(self, x, target_start, target_end):
        x = self.embedding(x)
        x = self.dropout(x)

        x, _ = self.bilstm(x)
        x = torch.squeeze(x, 0)

        start = target_start.data[0]
        end = target_end.data[0]

        if self.args.cuda:
            indices_left = None
            if not start == 0:
                indices_left = Variable(torch.cuda.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.cuda.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if not end == x.size(0) - 1:
                indices_right = Variable(torch.cuda.LongTensor([i for i in range(end + 1, x.size(0))]))
        else:
            indices_left = None
            if not start == 0:
                indices_left = Variable(torch.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if not end == x.size(0) - 1:
                indices_right = Variable(torch.LongTensor([i for i in range(end + 1, x.size(0))]))

        left = None
        if indices_left is not None:
            left = torch.index_select(x, 0, indices_left)
        target = torch.index_select(x, 0, indices_target)
        average_target = torch.mul(torch.sum(target, 0), 1 / (end - start + 1))
        average_target = torch.unsqueeze(average_target, 0)
        right = None
        if indices_right is not None:
            right = torch.index_select(x, 0, indices_right)

        s = self.attention(x, average_target)
        s_l = None
        if left is not None:
            s_l = self.attention_l(left, average_target)
        s_r = None
        if right is not None:
            s_r = self.attention_r(right, average_target)

        z = torch.exp(torch.mm(self.w1, torch.t(s)) + torch.mm(self.u1, torch.t(average_target)) + self.b1)
        if s_l is not None:
            z_l = torch.exp(torch.mm(self.w2, torch.t(s_l)) + torch.mm(self.u2, torch.t(average_target)) + self.b2)
        if s_r is not None:
            z_r = torch.exp(torch.mm(self.w3, torch.t(s_r)) + torch.mm(self.u3, torch.t(average_target)) + self.b3)

        ss = torch.mul(torch.t(z), s)
        if left is not None:
             ss = torch.add(ss, torch.mul(torch.t(z_l), s_l))
        if right is not None:
             ss = torch.add(ss, torch.mul(torch.t(z_r), s_r))

        result = self.linear_2(ss)
        return result


class AttentionContextGatedBiLSTM(nn.Module):
    def __init__(self, args, m_embedding):
        super(AttentionContextGatedBiLSTM, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embedding.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.bilstm = nn.LSTM(args.embed_dim, args.hidden_size,
                              dropout=args.dropout_rnn,
                              batch_first=True,
                              bidirectional=True)

        self.attention = Att(args.hidden_size * 2, args.attention_size, args.cuda)
        self.attention_l = Att(args.hidden_size * 2, args.attention_size, args.cuda)
        self.attention_r = Att(args.hidden_size * 2, args.attention_size, args.cuda)

        if args.cuda:
            self.w1 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2).cuda(), requires_grad=True)
            self.w2 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2).cuda(), requires_grad=True)
            self.w3 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2).cuda(), requires_grad=True)
            self.u1 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2).cuda(), requires_grad=True)
            self.u2 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2).cuda(), requires_grad=True)
            self.u3 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2).cuda(), requires_grad=True)
            self.b1 = Parameter(torch.randn(args.hidden_size * 2, 1).cuda(), requires_grad=True)
            self.b2 = Parameter(torch.randn(args.hidden_size * 2, 1).cuda(), requires_grad=True)
            self.b3 = Parameter(torch.randn(args.hidden_size * 2, 1).cuda(), requires_grad=True)
        else:
            self.w1 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2), requires_grad=True)
            self.w2 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2), requires_grad=True)
            self.w3 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2), requires_grad=True)
            self.u1 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2), requires_grad=True)
            self.u2 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2), requires_grad=True)
            self.u3 = Parameter(torch.randn(args.hidden_size * 2, args.hidden_size * 2), requires_grad=True)
            self.b1 = Parameter(torch.randn(args.hidden_size * 2, 1), requires_grad=True)
            self.b2 = Parameter(torch.randn(args.hidden_size * 2, 1), requires_grad=True)
            self.b3 = Parameter(torch.randn(args.hidden_size * 2, 1), requires_grad=True)

        nn.init.xavier_uniform(self.w1)
        nn.init.xavier_uniform(self.w2)
        nn.init.xavier_uniform(self.w3)
        nn.init.xavier_uniform(self.u1)
        nn.init.xavier_uniform(self.u2)
        nn.init.xavier_uniform(self.u3)
        nn.init.xavier_uniform(self.b1)
        nn.init.xavier_uniform(self.b2)
        nn.init.xavier_uniform(self.b3)

        self.linear_2 = nn.Linear(args.hidden_size * 2, args.label_num, bias=True)
        nn.init.xavier_uniform(self.linear_2.weight)

    def forward(self, x, target_start, target_end):
        x = self.embedding(x)
        x = self.dropout(x)

        x, _ = self.bilstm(x)
        x = torch.squeeze(x, 0)

        start = target_start.data[0]
        end = target_end.data[0]

        if self.args.cuda:
            indices_left = None
            if not start == 0:
                indices_left = Variable(torch.cuda.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.cuda.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if not end == x.size(0) - 1:
                indices_right = Variable(torch.cuda.LongTensor([i for i in range(end + 1, x.size(0))]))
        else:
            indices_left = None
            if not start == 0:
                indices_left = Variable(torch.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if not end == x.size(0) - 1:
                indices_right = Variable(torch.LongTensor([i for i in range(end + 1, x.size(0))]))

        left = None
        if indices_left is not None:
            left = torch.index_select(x, 0, indices_left)
        target = torch.index_select(x, 0, indices_target)
        average_target = torch.mul(torch.sum(target, 0), 1 / (end - start + 1))
        average_target = torch.unsqueeze(average_target, 0)
        right = None
        if indices_right is not None:
            right = torch.index_select(x, 0, indices_right)

        s = self.attention(x, average_target)
        s_l = None
        if left is not None:
            s_l = self.attention_l(left, average_target)
        s_r = None
        if right is not None:
            s_r = self.attention_r(right, average_target)

        z = torch.exp(torch.mm(self.w1, torch.t(s)) + torch.mm(self.u1, torch.t(average_target)) + self.b1)
        if s_l is not None:
            z_l = torch.exp(torch.mm(self.w2, torch.t(s_l)) + torch.mm(self.u2, torch.t(average_target)) + self.b2)
        if s_r is not None:
            z_r = torch.exp(torch.mm(self.w3, torch.t(s_r)) + torch.mm(self.u3, torch.t(average_target)) + self.b3)

        ss = torch.mul(torch.t(z), s)
        if left is not None:
             ss = torch.add(ss, torch.mul(torch.t(z_l), s_l))
        if right is not None:
             ss = torch.add(ss, torch.mul(torch.t(z_r), s_r))

        result = self.linear_2(ss)
        return result
