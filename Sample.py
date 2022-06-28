import json
import time
import importlib
import argparse
from collections import OrderedDict
import torch
from dataset import TestDataset
from util import *
import torch.nn as nn
from torchsummaryX import summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default='OverNet')
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--sample_dir", type=str, default='sample')
    parser.add_argument("--test_data_dir", type=str, default="dataset/Set5")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--num_gpu", type=int, default=3)

    return parser.parse_args()

def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def sample(net, dataset, cfg):
    avg_psnr = 0
    avg_ssim = 0

    total = 0
    cuda = True if torch.cuda.is_available() else False
    scale = cfg.scale
    for step, (hr, lr, name) in enumerate(dataset):

        t1 = time.time()
        h, w = lr.size()[1:]
        h_half, w_half = int(h / 2), int(w / 2)
        h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

        # split large image to 4 patch to avoid OOM error
        lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
        lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
        lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
        lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
        lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
        lr_patch = lr_patch.cuda()

        # run refine process in here!
        with torch.no_grad():
            sr = net(lr_patch, cfg.scale, cfg.upscale)

        h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
        w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

        # merge splited patch images
        result = torch.FloatTensor(3, h, w).cuda()
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
        result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
        result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
        sr = result

        t2 = time.time()


        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir, model_name,
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "SR")

        hr_dir = os.path.join(cfg.sample_dir,model_name, cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "HR")

        os.makedirs(sr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))

        save_image(sr, sr_im_path)
        save_image(hr, hr_im_path)

        sr = sr.unsqueeze(0).cuda()
        hr = hr.unsqueeze(0).cuda()

        psnr = calc_psnr(sr, hr, scale, 1, benchmark=True)
        avg_psnr += psnr / len(dataset)

        ti = t2 - t1
        print("Saved {} ({}x{} -> {}x{}, {:.3f}s -- PSNR {} )"
              .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2 - t1, psnr))
    print('Average PSNR on scale X{} is {} '.format(cfg.scale, avg_psnr))



def main(cfg):
    module = importlib.import_module("{}".format(cfg.model))
    net = module.Network(scale=cfg.scale,upscale=cfg.upscale,group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    state_dict = torch.load(cfg.ckpt_path)
    net.load_state_dict(state_dict['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # inputs = torch.zeros((1, 3, 1280//cfg.scale, 720//cfg.scale)).cuda()
    # summary(net, inputs, cfg.scale, cfg.upscale)

    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, dataset, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
