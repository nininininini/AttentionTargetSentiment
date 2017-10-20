# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


def train(args, model, train_iter, test_iter):
    model.train()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-6)

    m_max = -1
    whichmax = ''
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    output = open(os.path.join(args.save_dir, 'test.log'), "w+", encoding='utf-8')
    for attr, value in sorted(args.__dict__.items()):
        output.write("\t{}={} \n".format(attr.upper(), value))
        output.flush()
    output.write('----------------------------------------------------')
    output.flush()

    if args.lr_scheduler is not None:
        scheduler = None
        if args.lr_scheduler == 'lambda':
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.97 ** epoch
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda2)
        elif args.lr_scheduler == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == '':
            pass

    step = 0
    for epoch in range(1, args.epochs + 1):

        if args.lr_scheduler is not None:
            scheduler.step()
            print(scheduler.get_lr())
        print("第", epoch, "次迭代")

        for batch in train_iter:
            feature, target = batch.text, batch.label
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature, batch.target_start, batch.target_end)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  '
                                 'acc: {:.4f}%({}/{})'.format(step,
                                                              loss.data[0],
                                                              accuracy,
                                                              corrects,
                                                              batch.batch_size))

            if step % args.test_interval == 0:
                evaluate(args, model, test_iter)

            if step % args.save_interval == 0:
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_step{}.pt'.format(save_prefix, step)
                torch.save(model, save_path)

                m_str, acc = test(args, model, test_iter)
                output.write(m_str + '-------' + str(step) + '\n')
                output.flush()
                if acc > m_max:
                    m_max = acc
                    whichmax = step
    output.write('\nmax is {} using {}'.format(m_max, whichmax))
    output.flush()
    output.close()


def evaluate(args, model, iterator):
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for batch in iterator:
        feature, target = batch.text, batch.label
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature, batch.target_start, batch.target_end)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        size += batch.batch_size
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))


def test(args, model, iterator):
    model.eval()
    corrects, avg_loss = 0, 0
    size = 0
    for batch in iterator:
        feature, target = batch.text, batch.label
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature, batch.target_start, batch.target_end)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        size += batch.batch_size
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    return '\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size), accuracy


def predict():
    print('over-------')





















