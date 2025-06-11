# ood_eval_utils.py
import torch
import torch.nn.functional as F
import numpy as np

from config import net, args, ood_num_examples, concat, to_np, get_measures, print_measures

# this function is provided by DAL authors
# 1. for any input data loader, returns softmax scores
def dal_get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and not in_dist:
                break
            data, target = data.cuda(), target.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            _score.append(-np.max(smax, axis=1))
    if in_dist:
        return concat(_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

# this function is provided by DAL authors
# balance the number of in-dist data & ood data
def dal_get_and_print_results(ood_loader, in_score, num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = dal_get_ood_scores(ood_loader)
        if args.out_as_pos:
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr

# function for ID classification part
def eval_on_id_dataset():
    net.eval()
    correct = 0
    y, c = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(test_loader.dataset) * 100

# implement an eval function to evaluate the ID performance based on the result of `dal_get_ood_scores` (input as ID dataset)