#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net

import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    parser.add_argument('--dataset', default="/home2/pytorch-broad-models/F3Net", type=str)
    parser.add_argument('--ckpt', default="/home2/pytorch-broad-models/F3Net/model-32", type=str)
    args = parser.parse_args()
    print(args)
    return args

class Test(object):
    def __init__(self, args, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot=args.ckpt, mode='test')
        self.data   = Dataset.Data(self.cfg)

        # 1000 image
        self.loader = DataLoader(self.data, batch_size=args.batch_size, shuffle=False, num_workers=1)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)
                out = out2u

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()
    
    def save(self, args):
        total_time = 0.0
        total_sample = 0

        for i, (image, mask, shape, name) in enumerate(self.loader):
            if i >= args.num_iter:
                break
            image = image.cuda().float()
            elapsed = time.time()
            out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
            elapsed = time.time() - elapsed
            out   = out2u
            pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
            head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
            if not os.path.exists(head):
                os.makedirs(head)
            cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
        latency = total_time / total_sample * 1000
        throughput = total_sample / total_time
        print("inference Latency: {} ms".format(latency))
        print("inference Throughput: {} samples/s".format(throughput))

                


if __name__=='__main__':
    args = parse_args()

    for path in [args.dataset]:
        t = Test(args, dataset, F3Net, path)
        with torch.no_grad():
            t.save(args)
        # t.show()
