# -*- coding: utf-8 -*-

import torch

import os
import random
import datetime
import argparse

import model
import train
import dataset

random.seed(66)
torch.manual_seed(66)

parser = argparse.ArgumentParser(description='classificer')
# common
parser.add_argument('-epochs', type=int, default=64)
parser.add_argument('-batch-size', type=int, default=1)
parser.add_argument('-log-interval', type=int, default=1)
parser.add_argument('-test-interval', type=int, default=1000)
parser.add_argument('-save-interval', type=int, default=1000)
parser.add_argument('-save-dir', type=str, default='snapshot')
# random
parser.add_argument('-shuffle', action='store_true', default=True)
# model
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-dropout-embed', type=float, default=0.5)
parser.add_argument('-dropout-rnn', type=float, default=0.5)
parser.add_argument('-label-num', type=int, default=3)
# rnn
parser.add_argument('-use-embedding', action='store_true', default=True)
parser.add_argument('-embed-dim', type=int, default=300)
parser.add_argument('-max-norm', type=float, default=None)
parser.add_argument('-hidden-size', type=int, default=150)
parser.add_argument('-attention-size', type=int, default=200)
parser.add_argument('-weight-decay', type=int, default=0)
# device
parser.add_argument('-device', type=int, default=-1)
parser.add_argument('-cuda', action='store_true', default=False)
# option
parser.add_argument('-snapshot', type=str, default=None)
parser.add_argument('-predict', type=str, default=None)
parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-lr-scheduler', type=str, default=None)
parser.add_argument('-clip-norm', type=str, default=None)
parser.add_argument('-which-model', type=str, default='attentioncontext')

#
parser.add_argument('-need-smallembed', action='store_true', default=False)
parser.add_argument('-which-embedding', type=str, default='300d')
parser.add_argument('-which-data', type=str, default='Z')

args = parser.parse_args()

# load data and embedding
print("\nLoading data...")
if args.which_data == 'Z':
    train_path = './data/Z_data/all.conll.train'
    train_data = dataset.MyDatasets(args, train_path)
    # dev_path = ''
    # dev_data = dataset.MyDatasets(args, dev_path, train=train_data)
    test_path = './data/Z_data/all.conll.test'
    test_data = dataset.MyDatasets(args, test_path, train=train_data)
elif args.which_data == 'T':
    train_path = './data/T_data/train.conll'
    train_data = dataset.MyDatasets(args, train_path)
    test_path = './data/T_data/test.conll'
    test_data = dataset.MyDatasets(args, test_path, train=train_data)


# update args and print
args.label_num = len(train_data.vocabulary_label.word2id)
args.embed_num = len(train_data.vocabulary_text.word2id)
args.cuda = args.cuda and torch.cuda.is_available()
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model and cuda
m_model = None
if args.snapshot is None:
    if args.which_model == 'attention':
        m_model = model.Attention(args, train_data.embedding)
    elif args.which_model == 'attention2':
        m_model = model.Attention2(args, train_data.embedding)
    elif args.which_model == 'attention3':
        m_model = model.Attention3(args, train_data.embedding)
    elif args.which_model == 'attention4':
        m_model = model.Attention4(args, train_data.embedding)
    elif args.which_model == 'attentioncontext':
        m_model = model.AttentionContext(args, train_data.embedding)
    elif args.which_model == 'attentioncontextbilstm':
        m_model = model.AttentionContextBiLSTM(args, train_data.embedding)
    elif args.which_model == 'attentioncontextgated':
        m_model = model.AttentionContextGated(args, train_data.embedding)
    elif args.which_model == 'attentioncontextgatedbilstm':
        m_model = model.AttentionContextGatedBiLSTM(args, train_data.embedding)
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        m_model = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist.")
        exit()
if args.cuda:
    m_model.cuda()


# train and predict
torch.set_num_threads(1)
if args.predict is not None:
    pass
elif args.test:
    pass
else:
    train.train(args, m_model, train_data.iterator, test_data.iterator)
