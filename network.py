import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, UpsamplingBilinear2d, AvgPool2d, LeakyReLU, Flatten, LayerNorm
from torch.nn import Module, ModuleList, Sequential
from torch.optim import Adam
from random import randint

TOP_K = 16
SEFA_BOUNDARY = 2
BETAS = (0, 0.99)
#constant
LEAKY_RELU_GAIN = np.sqrt(2/(1+0.2**2))
ROOT_2 = 1.41421
ln2 = 0.69314

def make_noise_img(size):
    noise = np.random.normal(loc=0, scale=1.0, size=size)
    return torch.as_tensor(noise, dtype=torch.float32)

def init_weights(m):
    if type(m) == Linear or type(m) == Conv2d or type(m) == _ModulatedConv:
        torch.nn.init.normal_(m.weight)
        #torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class Weight_Scaling(Module):
    def __init__(self, fan_in, gain):
        super().__init__()
        self.kaiming_const = float(gain/np.sqrt(fan_in))

    def forward(self, x):
        #return x
        return self.kaiming_const*x

class Disc_Conv(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_scaling_1 = Weight_Scaling(in_channels*3*3, LEAKY_RELU_GAIN)
        self.conv_1 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU_1 = LeakyReLU(0.2)
        self.weight_scaling_2 = Weight_Scaling(in_channels*3*3, LEAKY_RELU_GAIN)
        self.conv_2 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU_2 = LeakyReLU(0.2)
        self.avgpool2d = AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.weight_scaling_1(x)
        x = self.conv_1(x)
        x = self.LeakyReLU_1(x)
        x = self.weight_scaling_2(x)
        x = self.conv_2(x)
        x = self.LeakyReLU_2(x)
        x = self.avgpool2d(x)
        return x

class Minibatch_Stddev(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #x: (N, C, H, W) torch.FloatTensor
        assert len(x.size()) == 4
        batch_size = x.size(0)
        H = x.size(2)
        W = x.size(3)
        stat = torch.std(x, dim=0)
        #(C, H, W)
        stat = torch.mean(stat, dim=0)
        #(H, W)
        stat = stat.repeat(batch_size, 1, 1, 1)
        x = torch.cat((x, stat), dim=1)
        return x


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.disc_block = Disc_Conv(in_channels, out_channels)
        self.weight_scaling = Weight_Scaling(in_channels*3*3, 1)
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        t = x
        x = self.disc_block(x)
        t = t.contiguous()
        t = F.avg_pool2d(self.conv(self.weight_scaling(t)), kernel_size=2, stride=2)
        x = (x + t)/ROOT_2
        return x

class Discriminator(Module):
    def __init__(self, disc_first_channel, disc_last_size, disc_lr, img_size, device):
        super().__init__()
        if not(device == 'cpu' or 'cuda:' in device):
            assert Exception('invalid argument in Network2.Discriminator')
        self.module_list = ModuleList()
        in_channels = 3
        out_channels = disc_first_channel
        #fromRGB
        self.module_list.append(Weight_Scaling(in_channels*1*1, LEAKY_RELU_GAIN))
        self.module_list.append(Conv2d(in_channels=in_channels, out_channels=disc_first_channel, kernel_size=1, stride=1))
        self.module_list.append(LeakyReLU(0.2))
        in_size = img_size
        cnt = 0
        while True:
            cnt += 1
            in_channels = out_channels
            out_channels *= 2
            if in_size == disc_last_size:
                break
            self.module_list.append(ResidualBlock(in_channels, out_channels))
            in_size //= 2
        self.module_list.append(Minibatch_Stddev())
        self.module_list.append(Weight_Scaling((in_channels+1)*4*4, LEAKY_RELU_GAIN))
        self.module_list.append(Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=4, stride=1, padding=0))
        self.module_list.append(LeakyReLU(0.2))
        self.module_list.append(Flatten())
        self.module_list.append(Weight_Scaling(in_channels, 1))
        self.module_list.append(Linear(in_channels, 1))
        self.to(device)
        self.opt = Adam(self.parameters(), lr=disc_lr, betas=BETAS)
        self.apply(init_weights)

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
            if (x != x).any():
                print('NaN occur!')
                assert False
        return x


class StyleMapper(Module):
    def __init__(self, z_size, style_size, mapping_lr, device):
        super().__init__()
        if not(device == 'cpu' or 'cuda:' in device):
            assert Exception('invalid argument in Network1.StyleMapper')
        self.styleblock = Sequential(
            Weight_Scaling(z_size, LEAKY_RELU_GAIN),
            Linear(z_size, style_size),
            LeakyReLU(),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
            LeakyReLU(0.2),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
            LeakyReLU(0.2),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
            LeakyReLU(0.2),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
            LeakyReLU(0.2),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
            LeakyReLU(0.2),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
            LeakyReLU(0.2),

            Weight_Scaling(style_size, LEAKY_RELU_GAIN),
            Linear(style_size, style_size),
        )
        self.to(device)
        self.opt = Adam(self.parameters(), lr=mapping_lr, betas=BETAS)
        self.apply(init_weights)


    def forward(self, z):
        z /= torch.sqrt(torch.mean(z**2, dim=1, keepdim=True) + 1e-8)
        style_base = self.styleblock(z)
        return style_base

class _ModulatedConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight_scale = Weight_Scaling(in_channels*kernel_size*kernel_size, LEAKY_RELU_GAIN)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.LeakyReLU = LeakyReLU(0.2)
        #[Cout,Cin,k,k]

    def forward(self, x, style_std, noise):
        # x: [N,Cin,H,W]
        # style_std: [N,Cin]
        #for equalized learning rate
        #x = self.weight_scale(x)
        batch_size = x.size(0)
        in_channels = x.size(1)
        out_channels = self.weight.size(0)
        H = x.size(2)
        W = x.size(3)
        weight = self.weight.view(1, self.weight.size(0), self.weight.size(1), self.weight.size(2), self.weight.size(3))
        #[1,Cout,Cin,k,k]*[batch,1,Cin,1,1]
        weight = weight*(style_std.view(style_std.size(0), 1, style_std.size(1), 1, 1))
        #[batch,Cout,Cin,k,k]
        weight_l2 = torch.sqrt(torch.sum(weight**2, dim=(2,3,4), keepdim=True)+1e-8)
        weight = weight/weight_l2
        weight = weight.view(-1,weight.size(2), weight.size(3), weight.size(4))
        #[batch*Cout,Cin,H,W]
        x = x.view(1, -1, x.size(2), x.size(3))
        #[1,N*C,H,W]
        padding_size = (self.weight.size(3)-1)//2
        x = F.conv2d(x, weight, groups=batch_size, padding=padding_size)
        x = x.view(batch_size, out_channels, H, W) + self.bias
        x += noise
        x = self.LeakyReLU(x)
        return x

class ModulatedConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_size, use_gpu, up, out):
        #up: upsamplex2?
        #out: RGB out?  
        super().__init__()
        self.use_gpu = use_gpu
        self.up = up
        self.style_scaling = Weight_Scaling(style_size, 1)
        self.style_affine = Linear(style_size, in_channels)
        self.modulated_conv = _ModulatedConv(in_channels, out_channels, kernel_size)
        self.noise_scalar = torch.nn.Parameter(torch.zeros(out_channels).view(1, out_channels, 1, 1))
        if out:
            self.name = 'LATTER'
            self.out = True
            self.out_weight_scale = Weight_Scaling(out_channels*1*1, 1)
            self.out_conv = Conv2d(out_channels, 3, 1)
        else:
            self.name = 'FORMER'
            self.out = False

    def forward(self, x, style_base, t=None):
        #x: [N,C,H,W]
        #style_base: [N,STYLE_SIZE]
        #t: for 'LATTER' block, residual connection!
        batch_size = x.size(0)
        style_std = self.style_affine(self.style_scaling(style_base))+1
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        img_size = x.size(2)
        noise = make_noise_img((batch_size, 1, img_size, img_size))
        if self.use_gpu:
            with torch.cuda.device_of(x):
                noise = noise.cuda()
        else:
            noise = noise.cpu()
        x = self.modulated_conv(x, style_std, self.noise_scalar*noise)
        if self.out:
            x = x.contiguous()
            if t is not None:
                x += t
                x /= ROOT_2
            out = self.out_conv(self.out_weight_scale(x))
            return x, out
        else:
            return x

class Generator(Module):
    def __init__(self, gen_channel, texture_size, style_size, style_mix_rate, gen_lr, img_size, device):
        super().__init__()
        #for sefa
        self.mae_eigen = None
        self.ato_eigen = None
        if device == 'cpu':
            use_gpu = False
        elif 'cuda' in device:
            use_gpu = True
        else:
            assert Exception('invalid argument in Network2.Generator')
        self.img_size = img_size
        self.style_mix_rate = style_mix_rate
        self.basic_texture = torch.nn.Parameter(torch.normal(torch.zeros(gen_channel, texture_size, texture_size), 1.0))
        self.module_list = ModuleList()
        self.conv1x1_list = ModuleList()
        first_block = ModulatedConvBlock(gen_channel, gen_channel, 3, style_size, use_gpu, up=False, out=True)
        self.module_list.append(first_block)
        in_size = 2*texture_size
        in_channels = gen_channel
        cnt = 0
        while True:
            cnt += 1
            former = ModulatedConvBlock(in_channels, in_channels, 3, style_size, use_gpu, up=True, out=False)
            if cnt > 1:
                latter = ModulatedConvBlock(in_channels, in_channels//2, 3, style_size, use_gpu, up=False, out=True)
                conv1x1 = Conv2d(in_channels, in_channels//2, 1)
                out_channels = in_channels//2
            else:
                latter = ModulatedConvBlock(in_channels, in_channels, 3, style_size, use_gpu, up=False, out=True)
                conv1x1 = Conv2d(in_channels, in_channels, 1)
                out_channels = in_channels
            self.module_list.append(former)
            self.module_list.append(latter)
            self.conv1x1_list.append(conv1x1)
            in_size *= 2
            in_channels = out_channels
            if in_size > img_size:
                break
        self.to(device)
        self.opt = Adam(self.parameters(), lr=gen_lr, betas=BETAS)
        self.apply(init_weights)

    def forward(self, style_base):
        #IN: style_base [2*batch_size, style_size]
        img = None
        cnt = 0
        batch_size = int(style_base.size(0)/2)
        basic_texture = self.basic_texture.repeat(batch_size, 1, 1, 1)
        #Non_style_mixing pass
        normal_idx = int((1-self.style_mix_rate)*batch_size)
        x = basic_texture[:normal_idx]
        t = None
        # t is for residual connection between 'FORMER' block and 'LATTER' block
        for m in self.module_list:
            if m.name == 'FORMER':
                t = x
                t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
                t = self.conv1x1_list[cnt-1](t)
                #for equalized learning rate
                #gain = 1
                t /= float(np.sqrt(self.conv1x1_list[cnt-1].weight.size(1)))
                x = m(x, style_base[:normal_idx])
            elif m.name == 'LATTER':
                cnt += 1
                x, rgb = m(x, style_base[:normal_idx], t)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == self.img_size:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError(m.name,'in generator, unknown block name')
        #img /= cnt
        normal_img = F.tanh(img)
        #style_mixing pass
        img = None
        cnt = 0
        style_change_layer = randint(1,5)
        #style_change_layer = 1
        x = basic_texture[normal_idx:]
        t = None
        style_base1 = style_base[normal_idx:batch_size]
        style_base2 = style_base[batch_size:batch_size + len(style_base1)]
        for m in self.module_list:
            if m.name == 'FORMER':
                t = x
                t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
                t = self.conv1x1_list[cnt-1](t)
                #for equalized learning rate
                #gain = 1
                t /= float(np.sqrt(self.conv1x1_list[cnt-1].weight.size(1)))
                if cnt <= style_change_layer:
                    x = m(x, style_base1)
                else:
                    x = m(x, style_base2)
            elif m.name == 'LATTER':
                cnt += 1
                if cnt <= style_change_layer:
                    x, rgb = m(x, style_base1, t)
                else:
                    x, rgb = m(x, style_base2, t)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == self.img_size:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError(m.name,'in generator, unknown block name')
        #img /= cnt
        style_mix_img = F.tanh(img)
        img = torch.cat([normal_img, style_mix_img], dim=0)
        return img

    def generate(self, style_base, style_change_layer):
        #IN: style_base [2*batch_size, style_size]
        img = None
        cnt = 0
        batch_size = int(style_base.size(0)/2)
        basic_texture = self.basic_texture.repeat(batch_size, 1, 1, 1)
        #Non_style_mixing pass
        normal_idx = int((1-self.style_mix_rate)*batch_size)
        x = basic_texture[:normal_idx]
        t = None
        # t is for residual connection between 'FORMER' block and 'LATTER' block
        for m in self.module_list:
            if m.name == 'FORMER':
                t = x
                t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
                t = self.conv1x1_list[cnt-1](t)
                #for equalized learning rate
                #gain = 1
                t /= float(np.sqrt(self.conv1x1_list[cnt-1].weight.size(1)))
                x = m(x, style_base[:normal_idx])
            elif m.name == 'LATTER':
                cnt += 1
                x, rgb = m(x, style_base[:normal_idx], t)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == self.img_size:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError(m.name,'in generator, unknown block name')
        #img /= cnt
        normal_img = F.tanh(img)
        #style_mixing pass
        img = None
        cnt = 0
        #style_change_layer = randint(1,5)
        #style_change_layer = 1
        x = basic_texture[normal_idx:]
        t = None
        style_base1 = style_base[normal_idx:batch_size]
        style_base2 = style_base[batch_size:batch_size + len(style_base1)]
        for m in self.module_list:
            if m.name == 'FORMER':
                t = x
                t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
                t = self.conv1x1_list[cnt-1](t)
                #for equalized learning rate
                #gain = 1
                t /= float(np.sqrt(self.conv1x1_list[cnt-1].weight.size(1)))
                if cnt == style_change_layer:
                    x = m(x, style_base1)
                else:
                    x = m(x, style_base2)
            elif m.name == 'LATTER':
                cnt += 1
                if cnt == style_change_layer:
                    x, rgb = m(x, style_base1, t)
                else:
                    x, rgb = m(x, style_base2, t)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == self.img_size:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError(m.name,'in generator, unknown block name')
        #img /= cnt
        style_mix_img = F.tanh(img)
        img = torch.cat([normal_img, style_mix_img], dim=0)
        return img

    def sefa(self, style_base, sefa_altering_layer='mae', sefa_eigen_num=0, sefa_intensity=1):
        #IN: style_base : [batch_size, style_size], sefa_altering_layer : str in {mae, ato}, sefa_eigen_num : int < TOP_K
        if self.mae_eigen is None:
            print('calculating eigens')
            cnt = 0
            mae_list = []
            ato_list = []
            for m in self.module_list:
                if m.name == 'FORMER':
                    if cnt < SEFA_BOUNDARY:
                        mae_list.append(m.style_affine)
                    else:
                        ato_list.append(m.style_affine)
                elif m.name == 'LATTER':
                    if cnt < SEFA_BOUNDARY:
                        mae_list.append(m.style_affine)
                    else:
                        ato_list.append(m.style_affine)
                    cnt += 1
                else:
                    raise NotImplementedError(m.name,'in generator, unknown block name')
            tmp = None
            for m in mae_list:
                if tmp is None:
                    tmp = m.weight
                else:
                    tmp = torch.cat([tmp, m.weight], dim=0)
            mae_weights = tmp
            _, _, vt = torch.svd(mae_weights)
            mae_eigen = []
            for i in range(TOP_K):
                mae_eigen.append(vt[i, :].unsqueeze(0))
            self.mae_eigen = mae_eigen
            #######################################
            
            tmp = None
            for m in ato_list:
                if tmp is None:
                    tmp = m.weight
                else:
                    tmp = torch.cat([tmp, m.weight], dim=0)
            ato_weights = tmp
            _, _, vt = torch.svd(ato_weights)
            ato_eigen = []
            for i in range(TOP_K):
                ato_eigen.append(vt[i, :].unsqueeze(0))
            self.ato_eigen = ato_eigen
            
        ########################################
        cnt = 0
        img = None
        batch_size = int(style_base.size(0))
        basic_texture = self.basic_texture.repeat(batch_size, 1, 1, 1)
        x = basic_texture
        t = None
        # t is for residual connection between 'FORMER' block and 'LATTER' block
        for m in self.module_list:
            if m.name == 'FORMER':
                t = x
                t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
                t = self.conv1x1_list[cnt-1](t)
                #for equalized learning rate
                #gain = 1
                t /= float(np.sqrt(self.conv1x1_list[cnt-1].weight.size(1)))
                if sefa_eigen_num >= 0:
                    if (sefa_altering_layer == 'mae' or sefa_altering_layer == 'zenbu') and cnt < SEFA_BOUNDARY:
                        x = m(x, style_base + sefa_intensity * self.mae_eigen[sefa_eigen_num])
                    elif (sefa_altering_layer == 'ato' or sefa_altering_layer == 'zenbu') and cnt >= SEFA_BOUNDARY:
                        x = m(x, style_base + sefa_intensity * self.ato_eigen[sefa_eigen_num])
                    else:
                        x = m(x, style_base)
                else:
                    x = m(x, style_base)
            elif m.name == 'LATTER':
                cnt += 1
                if sefa_eigen_num >= 0:
                    if (sefa_altering_layer == 'mae' or sefa_altering_layer == 'zenbu') and cnt < SEFA_BOUNDARY:
                        x, rgb = m(x, style_base + sefa_intensity * self.mae_eigen[sefa_eigen_num], t)
                    elif (sefa_altering_layer == 'ato' or sefa_altering_layer == 'zenbu') and cnt >= SEFA_BOUNDARY:
                        x, rgb = m(x, style_base + sefa_intensity * self.ato_eigen[sefa_eigen_num], t)
                    else:
                        x, rgb = m(x, style_base, t)
                else:
                    x, rgb = m(x, style_base, t)
                if img is None:
                    img = rgb
                else:
                    img = img + rgb
                if x.size(2) == self.img_size:
                    #last layer doesn't need bilinear upsampling!
                    break
                img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError(m.name,'in generator, unknown block name')
        #img /= cnt
        img = F.tanh(img)
        return img
        


if __name__ == '__main__':
    print('testing Networkv2.py')
    
