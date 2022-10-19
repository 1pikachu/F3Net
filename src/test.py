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

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

class Test(object):
    def __init__(self, args, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot=args.ckpt, mode='test')
        self.data   = Dataset.Data(self.cfg)

        # 1000 image
        self.loader = DataLoader(self.data, batch_size=args.batch_size, shuffle=False, num_workers=1)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.to(args.device)

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
        if args.channels_last:
            try:
                self.net = self.net.to(memory_format=torch.channels_last)
                print("---- use NHWC format")
            except RuntimeError as e:
                print("---- use normal format")
                print("failed enable channels_last: ", e)
        if args.nv_fuser:
           fuser_mode = "fuser2"
        else:
           fuser_mode = "none"
        print("---- fuser mode:", fuser_mode)

        total_time = 0.0
        total_sample = 0
        profile_len = min(len(self.loader), args.num_iter + args.num_warmup) // 2

        if args.profile and args.device == "xpu":
            for i, (image, mask, shape, name) in enumerate(self.loader):
                if i >= args.num_iter:
                    break

                image = image.to(args.device).float()
                mask = mask.to(args.device)
                shape = [i.to(args.device) for i in shape]
                if args.channels_last:
                    image = image.to(memory_format=torch.channels_last) if len(image.shape) == 4 else image
                if args.jit and i == 0:
                    try:
                        self.net = torch.jit.trace(self.net, (image,shape), check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        self.net = torch.jit.freeze(self.net)
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                    elapsed = time.time()
                    out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                    torch.xpu.synchronize()
                    elapsed = time.time() - elapsed
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().float().numpy()
                head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
                if args.profile and i == profile_len:
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        try:
                            os.makedirs(timeline_dir)
                        except:
                            pass
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                        timeline_dir+'profile.pt')
                    torch.save(prof.key_averages(group_by_input_shape=True).table(),
                        timeline_dir+'profile_detail.pt')
        elif args.profile and args.device == "cuda":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_len,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i, (image, mask, shape, name) in enumerate(self.loader):
                    if i >= args.num_iter:
                        break

                    image = image.to(args.device).float()
                    mask = mask.to(args.device)
                    shape = [i.to(args.device) for i in shape]
                    if args.channels_last:
                        image = image.to(memory_format=torch.channels_last) if len(image.shape) == 4 else image
                    if args.jit and i == 0:
                        try:
                            self.net = torch.jit.trace(self.net, (image,shape), check_trace=False, strict=False)
                            print("---- JIT trace enable.")
                            self.net = torch.jit.freeze(self.net)
                        except (RuntimeError, TypeError) as e:
                            print("---- JIT trace disable.")
                            print("failed to use PyTorch jit mode due to: ", e)
                        
                    elapsed = time.time()
                    with torch.jit.fuser(fuser_mode):
                        out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                    torch.cuda.synchronize()
                    elapsed = time.time() - elapsed
                    p.step()
                    out   = out2u
                    pred  = (torch.sigmoid(out[0,0])*255).cpu().float().numpy()
                    head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
                    if not os.path.exists(head):
                        os.makedirs(head)
                    cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_sample += args.batch_size
                        total_time += elapsed
        elif args.profile and args.device == "cpu":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_len,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i, (image, mask, shape, name) in enumerate(self.loader):
                    if i >= args.num_iter:
                        break

                    image = image.to(args.device).float()
                    mask = mask.to(args.device)
                    shape = [i.to(args.device) for i in shape]
                    if args.channels_last:
                        image = image.to(memory_format=torch.channels_last) if len(image.shape) == 4 else image
                    if args.jit and i == 0:
                        try:
                            self.net = torch.jit.trace(self.net, (image,shape), check_trace=False, strict=False)
                            print("---- JIT trace enable.")
                            self.net = torch.jit.freeze(self.net)
                        except (RuntimeError, TypeError) as e:
                            print("---- JIT trace disable.")
                            print("failed to use PyTorch jit mode due to: ", e)
                        
                    elapsed = time.time()
                    out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                    elapsed = time.time() - elapsed
                    p.step()
                    out   = out2u
                    pred  = (torch.sigmoid(out[0,0])*255).cpu().float().numpy()
                    head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
                    if not os.path.exists(head):
                        os.makedirs(head)
                    cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_sample += args.batch_size
                        total_time += elapsed
        elif not args.profile and args.device == "cuda":
            for i, (image, mask, shape, name) in enumerate(self.loader):
                if i >= args.num_iter:
                    break

                image = image.to(args.device).float()
                mask = mask.to(args.device)
                shape = [i.to(args.device) for i in shape]
                if args.channels_last:
                    image = image.to(memory_format=torch.channels_last) if len(image.shape) == 4 else image
                if args.jit and i == 0:
                    try:
                        self.net = torch.jit.trace(self.net, (image,shape), check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        self.net = torch.jit.freeze(self.net)
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                    
                elapsed = time.time()
                with torch.jit.fuser(fuser_mode):
                    out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().float().numpy()
                head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
        else:
            for i, (image, mask, shape, name) in enumerate(self.loader):
                if i >= args.num_iter:
                    break

                image = image.to(args.device).float()
                mask = mask.to(args.device)
                shape = [i.to(args.device) for i in shape]
                if args.channels_last:
                    image = image.to(memory_format=torch.channels_last) if len(image.shape) == 4 else image
                if args.jit and i == 0:
                    try:
                        self.net = torch.jit.trace(self.net, (image,shape), check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                        self.net = torch.jit.freeze(self.net)
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
                    
                elapsed = time.time()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                if args.device == "xpu":
                    torch.xpu.synchronize()
                elapsed = time.time() - elapsed
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().float().numpy()
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

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    t = Test(args, dataset, F3Net, args.dataset)
    with torch.no_grad():
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                t.save(args)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                t.save(args)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                t.save(args)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                t.save(args)
        else:
            print("---- no autocast")
            t.save(args)
