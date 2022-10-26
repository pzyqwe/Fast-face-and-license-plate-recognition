# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from util.general import colorstr


def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1. / thr).float().sum(1).mean()
        bpr = (best > 1. / thr).float().mean()
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
    print('')


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    thr = 1. / thr
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]
        return x, x.max(1)[0]

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k):
        k = k[np.argsort(k.prod(1))]
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)
    k, dist = kmeans(wh / s, n, iter=30)
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)
    wh0 = torch.tensor(wh0, dtype=torch.float32)
    k = print_results(k)
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)
