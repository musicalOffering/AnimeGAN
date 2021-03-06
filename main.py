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
from torch.nn import Module
from torch.nn.parallel import DataParallel
from CustomDataset import TANOCIv2_Dataset

#constant
ROOT_2 = 1.41421
ln2 = 0.69314

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def averaging_param(old_params:list, current_params:list, tau:float=0.99):
    #old_param should be averaging network
    for old_param, current_param in zip(old_params, current_params):
        old_param.data = (current_param.data * (1.0 - tau) + old_param.data * tau).clone()
    
def cp_module(src:Module, dst:Module):
    for src_param, dst_param in zip(src.parameters(), dst.parameters()):
        dst_param.data = src_param.data.clone()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #general argument
    parser.add_argument('--load_index', default=0)
    parser.add_argument('--device', default='multi')
    parser.add_argument('--dataset_path', default='./TANOCI-v2')
    parser.add_argument('--save_path', default='./result')
    parser.add_argument('--epoch', default=1000)
    parser.add_argument('--img_level', default=7)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--verbose_freq', default=100)
    parser.add_argument('--save_freq', default=10)
    parser.add_argument('--z_cov', default=1)
    parser.add_argument('--ema_decay', default=0.99)
    parser.add_argument('--ema_start', default=1e15)

    #generator arguments
    parser.add_argument('--style_size', default=512)
    parser.add_argument('--style_mix_rate', default=0.5)
    parser.add_argument('--z_size', default=512)
    parser.add_argument('--texture_size', default=4)
    parser.add_argument('--gen_channel', default=768)
    parser.add_argument('--gen_lr', default=0.001)
    parser.add_argument('--mapping_lr_ratio', default=0.01)
    parser.add_argument('--gen_lazy', default=8)

    #discriminator arguments
    parser.add_argument('--disc_first_channel', default=32)
    parser.add_argument('--disc_last_size', default=4)
    parser.add_argument('--disc_lr', default=0.002)
    parser.add_argument('--gp_coef', default=10)
    args = parser.parse_args()

    load_index = int(args.load_index)
    device = str(args.device)
    #device: 'multi', 'cpu', 'cuda:%d'
    #'multi' uses all the GPUs
    dataset_path = str(args.dataset_path)
    save_path = str(args.save_path)
    epoch = int(args.epoch)
    img_level = int(args.img_level)
    img_size = 2**img_level
    batch_size = int(args.batch_size)
    verbose_freq = int(args.verbose_freq)
    save_freq = int(args.save_freq)
    z_cov = float(args.z_cov)
    ema_decay = float(args.ema_decay)
    ema_start = int(args.ema_start)

    style_size = int(args.style_size)
    style_mix_rate = float(args.style_mix_rate)
    z_size = int(args.z_size)
    texture_size = int(args.texture_size)
    gen_channel = int(args.gen_channel)
    gen_lr = float(args.gen_lr)
    mapping_lr = float(args.mapping_lr_ratio)*gen_lr
    gen_lazy = int(args.gen_lazy)
    ema_coef = (gen_lazy*ln2)/((img_size**2)*(img_level-1)*ln2)

    disc_first_channel = int(args.disc_first_channel)
    disc_last_size = int(args.disc_last_size)
    disc_lr = float(args.disc_lr)
    gp_coef = float(args.gp_coef)

    if device == 'multi':
        device = 'cuda:0'
        use_multi_gpu = True
    elif device == 'cpu':
        use_multi_gpu = False
    elif 'cuda:' in device:
        use_multi_gpu = False
    else:
        raise Exception('invalid argument in device (main)')

    #basic transformation for our dataset
    basic_transform = T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(brightness=0, contrast=0.1, saturation=0.1)])
    dataset = TANOCIv2_Dataset(img_size=img_size, dataset_path=dataset_path, transform=basic_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    first = True
    S = network.StyleMapper(z_size, style_size, mapping_lr, device)
    G = network.Generator(gen_channel, texture_size, style_size, style_mix_rate, gen_lr, img_size, device)
    D = network.Discriminator(disc_first_channel, disc_last_size, disc_lr, img_size, device)
    G_average = network.Generator(gen_channel, texture_size, style_size, style_mix_rate, gen_lr, img_size, device)
    cp_module(G, G_average)


    if load_index != 0:
        print('loading', load_index, 'models...')
        S_load_path = save_path + '_weight/' + str(load_index) + 'S.pt'
        G_load_path = save_path + '_weight/' + str(load_index) + 'G.pt'
        D_load_path = save_path + '_weight/' + str(load_index) + 'D.pt'
        S.load_state_dict(torch.load(S_load_path))
        G.load_state_dict(torch.load(G_load_path))
        D.load_state_dict(torch.load(D_load_path))
        cp_module(G, G_average)
        print('loading complete!')
    else:
        print('Using ', torch.cuda.device_count(), 'GPUs...')
    print('Parameter numbers: S, G, D')
    print(count_parameters(S), count_parameters(G), count_parameters(D))
    #wrapping DataParallel
    S = DataParallel(S)
    G = DataParallel(G)
    G_average = DataParallel(G_average)
    D = DataParallel(D)
    #visual seed
    dist = MultivariateNormal(loc=torch.zeros(2*batch_size, z_size), covariance_matrix=z_cov*torch.eye(z_size))
    visual_seed = torch.FloatTensor(dist.sample()).to(device)
    #visual_seed = torch.rand(2*batch_size, z_size).to(device)

    previous_grads_norm = 0
    step_cnt = 1 + verbose_freq*load_index
    for e in range(epoch):
        for real in dataloader:
            real = real.to(device)
            real_batch_size = real.size()[0]
            z = torch.FloatTensor(dist.sample()).to(device)[:2*real_batch_size]
            #z = torch.rand(2*real_batch_size, z_size).to(device)
            print('------------!--------------!------------')
            #d_update
            G_fake = G(S(z))
            fake = G_fake.detach()
            real_out = D(real)
            fake_out = D(fake)
            d_loss = -(torch.mean(real_out) - torch.mean(fake_out))
            print('epoch:', e, 'step: ', step_cnt, 'd_loss', d_loss)
            #gradient penalty for WGAN_GP
            epsilon = torch.rand(real_batch_size, 1, 1, 1).to(device)
            interpolated = epsilon*fake + (1-epsilon)*real
            interpolated.requires_grad_()
            interpolated_out = D(interpolated)
            grads = torch.autograd.grad(interpolated_out, interpolated, 
                grad_outputs=torch.ones_like(interpolated_out).to(device), 
                retain_graph=True, create_graph=True
            )[0]
            grads = grads.view(real_batch_size, -1)
            grad_penalty = torch.mean((grads.norm(2, dim=1)-1)**2)
            print('WGAN grad_penalty: ', float(grad_penalty.detach().cpu().numpy()))
            d_loss += gp_coef*grad_penalty
            D.module.opt.zero_grad()  
            d_loss.backward()
            D.module.opt.step()
            #g_update
            if step_cnt <= ema_start:
                cp_module(G.module, G_average.module)
            elif step_cnt > ema_start:
                G_old_parmas = [p.clone().detach() for p in G_average.module.parameters()]
            fake_out = D(G_fake)
            g_loss = -torch.mean(fake_out)
            print('epoch:', e, 'step: ', step_cnt, '!!!g_loss', g_loss)
            '''
            if step_cnt % gen_lazy == 0:
                #path length Regulation
                y = torch.randn(real_batch_size, 3*img_size*img_size).to(device)
                y /= img_size
                w = S(z)
                Gw = G(w).view(real_batch_size, -1)
                Jy = torch.bmm(Gw.view(real_batch_size, 1, -1), y.view(real_batch_size, -1, 1))
                grads = torch.autograd.grad(Jy, w, 
                    grad_outputs=torch.ones_like(Jy).to(device),
                    retain_graph=True, create_graph=True
                )[0]
                grads = grads.view(real_batch_size, -1)
                grads_norm = grads.norm(2, dim=1)
                current_grads_norm = float(torch.mean(grads_norm).detach().cpu().numpy())
                grad_penalty = torch.mean(grads_norm - previous_grads_norm)**2
                print('gen_grad_penalty', float(grad_penalty.detach().cpu().numpy()))
                previous_grads_norm = previous_grads_norm*ema_decay + (1-ema_decay)*current_grads_norm
                print('a:', previous_grads_norm)
                g_loss += ema_coef*grad_penalty
            '''                  
            S.module.opt.zero_grad()
            G.module.opt.zero_grad()
            g_loss.backward()
            S.module.opt.step()
            G.module.opt.step()
            if step_cnt > ema_start:
                print('step ', step_cnt, 'SG averaging param applied...')
                G_current_params = [p for p in G.module.parameters()]
                averaging_param(G_old_parmas, G_current_params)
            if first or step_cnt == 1 or step_cnt % verbose_freq == 0:
                first = False
                img_save_path = save_path + '_img/'
                index = str(step_cnt//verbose_freq)
                vis_fake = G_average(S(visual_seed)).detach()
                save_image(make_grid(vis_fake), img_save_path + index + '.jpg', normalize=True)
                save_image(make_grid(vis_fake), 'tmp.jpg', normalize=True)
                if int(index) % save_freq == 0:
                    weight_save_path = save_path + '_weight/'
                    torch.save(S.module.state_dict(), weight_save_path + index + 'S.pt')
                    torch.save(G_average.module.state_dict(), weight_save_path + index +  'G.pt')
                    torch.save(D.module.state_dict(), weight_save_path + index + 'D.pt')
            step_cnt += 1