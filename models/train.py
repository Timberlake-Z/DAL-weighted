# train.py
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging

def train_DAL(epoch, gamma, net, optimizer, scheduler, train_loader_in, train_loader_out, args):
    net.train()
    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))

    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        x, emb = net.pred_emb(data)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        l_oe_old = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        emb_oe = emb[len(in_set[0]):].detach()
        emb_bias = torch.rand_like(emb_oe) * 0.0001

        for _ in range(args.iter):
            emb_bias.requires_grad_()
            x_aug = net.fc(emb_bias + emb_oe)
            l_sur = - (x_aug.mean(1) - torch.logsumexp(x_aug, dim=1)).mean()
            r_sur = (emb_bias.abs()).mean(-1).mean()
            l_sur = l_sur - r_sur * gamma
            grads = torch.autograd.grad(l_sur, [emb_bias])[0]
            grads /= (grads ** 2).sum(-1).sqrt().unsqueeze(1)
            emb_bias = emb_bias.detach() + args.strength * grads.detach()
            optimizer.zero_grad()
        
        gamma -= args.beta * (args.rho - r_sur.detach())
        gamma = gamma.clamp(min=0.0, max=args.gamma)

        if epoch >= args.warmup:
            x_oe = net.fc(emb[len(in_set[0]):] + emb_bias)
        else:
            x_oe = net.fc(emb[len(in_set[0]):])

        l_oe = - (x_oe.mean(1) - torch.logsumexp(x_oe, dim=1)).mean()
        loss = l_ce + .5 * l_oe

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' % (epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        logging.info(f"Epoch {epoch:2d} Batch {batch_idx+1}/{len(train_loader_in)} Loss {loss_avg:.4f}")
        scheduler.step()
    
    return gamma


def train_fw(epoch, gamma, net, optimizer, scheduler, train_loader_in, train_loader_out, args):

    
    net.train()
    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))

    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        x, emb = net.pred_emb(data)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        l_oe_old = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        emb_oe = emb[len(in_set[0]):].detach()
        emb_bias = torch.rand_like(emb_oe) * 0.0001

        for _ in range(args.iter):
            emb_bias.requires_grad_()
            x_aug = net.fc(emb_bias + emb_oe)
            l_sur = - (x_aug.mean(1) - torch.logsumexp(x_aug, dim=1)).mean()
            r_sur = (emb_bias.abs()).mean(-1).mean()
            l_sur = l_sur - r_sur * gamma
            grads = torch.autograd.grad(l_sur, [emb_bias])[0]
            grads /= (grads ** 2).sum(-1).sqrt().unsqueeze(1)
            emb_bias = emb_bias.detach() + args.strength * grads.detach()
            optimizer.zero_grad()
        
        gamma -= args.beta * (args.rho - r_sur.detach())
        gamma = gamma.clamp(min=0.0, max=args.gamma)

        if epoch >= args.warmup:
            x_oe = net.fc(emb[len(in_set[0]):] + emb_bias)
        else:
            x_oe = net.fc(emb[len(in_set[0]):])

        l_oe = - (x_oe.mean(1) - torch.logsumexp(x_oe, dim=1)).mean()
        loss = l_ce + .5 * l_oe

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' % (epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        logging.info(f"Epoch {epoch:2d} Batch {batch_idx+1}/{len(train_loader_in)} Loss {loss_avg:.4f}")
        scheduler.step()
    
    return gamma