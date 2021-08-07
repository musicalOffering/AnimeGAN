import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import network
from copy import deepcopy
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.nn import Module, DataParallel
from CustomDataset import TANOCIv2_Dataset

COMPUTE_MEAN_ITER = 10000
INTENSITY_CONST = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_num', required=True)
    parser.add_argument('--load_index', required=True)
    parser.add_argument('--load_path', default='./result')
    parser.add_argument('--save_path', default='./sefa/')
    parser.add_argument('--psi', default=0.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--img_level', default=7)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--z_cov', default=1)
    parser.add_argument('--style_size', default=512)
    parser.add_argument('--z_size', default=512)
    parser.add_argument('--texture_size', default=4)
    parser.add_argument('--gen_channel', default=768)
    parser.add_argument('--sefa_layer', default='mae')
    parser.add_argument('--eigen_num', default=-1)

    args = parser.parse_args()
    gen_num = int(args.gen_num)
    load_index = int(args.load_index)
    if load_index <= 0:
        raise Exception('You have to specify load_index')
    load_path = str(args.load_path)
    save_path = str(args.save_path)
    psi = float(args.psi)
    device = str(args.device)

    img_level = int(args.img_level)
    img_size = 2**img_level
    batch_size = int(args.batch_size)
    z_cov = float(args.z_cov)

    style_size = int(args.style_size)
    style_mix_rate = 0.5
    z_size = int(args.z_size)
    texture_size = int(args.texture_size)
    gen_channel = int(args.gen_channel)

    sefa_layer = str(args.sefa_layer)
    eigen_num = int(args.eigen_num)

    if device == 'multi':
        raise Exception('multi gpu not allowed in sampling!')
    elif device == 'cpu':
        use_multi_gpu = False
    elif 'cuda' in device:
        use_multi_gpu = False
    else:
        raise Exception('invalid argument in device (sampling)')

    S = network.StyleMapper(z_size, style_size, 0, device)
    G = network.Generator(gen_channel, texture_size, style_size, style_mix_rate, 0, img_size, device)

    print('loading', load_index, 'models...')
    S_load_path = load_path + '_weight/' + str(load_index) + 'S.pt'
    G_load_path = load_path + '_weight/' + str(load_index) + 'G.pt'
    S.load_state_dict(torch.load(S_load_path))
    G.load_state_dict(torch.load(G_load_path))
    dist = MultivariateNormal(loc=torch.zeros(batch_size, z_size), covariance_matrix=z_cov*torch.eye(z_size))

    tmp = None
    for _ in range(COMPUTE_MEAN_ITER):
        z = dist.sample()
        if 'cuda' in device:
            z = z.cuda()
        w = S(z)
        if tmp is None:
            tmp = w
        else:
            tmp += w
    style_mean = torch.mean(tmp, axis=0)/COMPUTE_MEAN_ITER
    for e in range(gen_num):
        z = dist.sample()
        if 'cuda' in device:
            z = z.cuda()
        w = S(z)
        w = style_mean + psi*(w - style_mean)
        for i in range(11):
            intensity = INTENSITY_CONST*(i-5)
            img = G.sefa(w, sefa_layer, eigen_num, intensity)
            img_save_path = f'{save_path}{e}{sefa_layer}{eigen_num}_{intensity}.jpg'
            save_image(make_grid(img), img_save_path, normalize=True)







