import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class AdaILN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1.1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        in_mean, in_var = \
            torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True), \
            torch.var(torch.var(x, dim=2, keepdim=True), dim=3, keepdim=True)

        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = \
            torch.mean(torch.mean(torch.mean(x, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True), \
            torch.var(torch.var(torch.var(x, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1.1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        in_mean, in_var = \
            torch.mean(x, dim=[2, 3], keepdim=True), \
            torch.var(x, dim=[2, 3], keepdim=True)

        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = \
            torch.mean(x, dim=[1, 2, 3], keepdim=True), \
            torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


class ResidualAdaLINBlock(nn.Module):
    """Residual Block w/ Adaptive Instance Layer Normalize"""

    def __init__(self, dim_in: int, dim_out: int, use_bias: bool = False):
        super(ResidualAdaLINBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = AdaILN(dim_out)
        self.relu1 = nn.LeakyReLU(.2, True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = AdaILN(dim_out)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class ResidualBlock(nn.Module):
    """Residual Block w/ Instance Normalize"""

    def __init__(self, dim_in: int, dim_out: int, use_bias: bool = False):
        super(ResidualBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = nn.InstanceNorm2d(dim_out)
        self.relu1 = nn.LeakyReLU(.2, True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = nn.InstanceNorm2d(dim_out)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + x


class ContentEncoder(nn.Module):
    """ContentEncoder Network w/ mask"""

    def __init__(self, conv_dim: int = 64, n_down_blocks: int = 2, n_res_blocks: int = 6):
        super(ContentEncoder, self).__init__()

        init_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.LeakyReLU(.2, True)
        ]

        layers = list()
        curr_dim: int = conv_dim
        for i in range(n_down_blocks):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.LeakyReLU(.2, True)
            ]
            curr_dim *= 2

        for i in range(n_res_blocks):
            layers += [
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, use_bias=False)
            ]

        self.gap_fc = nn.Linear(curr_dim, 1, bias=False)
        self.gmp_fc = nn.Linear(curr_dim, 1, bias=False)
        self.conv1x1 = nn.Conv2d(curr_dim * 2, curr_dim, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(.2, True)

        self.init = nn.Sequential(*init_layers)
        self.model = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x_init = self.init(x)
        if mask is not None:
            x_init = x_init * mask
        x_out = self.model(x_init)

        gap = F.adaptive_avg_pool2d(x_out, 1)
        gap_logit = self.gap_fc(gap.view(x_out.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x_out * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x_out, 1)
        gmp_logit = self.gmp_fc(gmp.view(x_out.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x_out * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.conv1x1(x)
        out = self.leaky_relu(x)
        return out, cam_logit


class StyleEncoder(nn.Module):
    """StyleEncoder Network"""

    def __init__(self, img_size: int = 128,
                 conv_dim: int = 64, n_down_blocks: int = 2, n_res_blocks: int = 3, lat_dim: int = 256):
        super(StyleEncoder, self).__init__()

        self.shared_style_encoder = ContentEncoder(
            conv_dim=conv_dim,
            n_down_blocks=n_down_blocks,
            n_res_blocks=n_res_blocks
        )

        curr_dim: int = conv_dim * (2 ** n_down_blocks)

        mlp_layers = [
            nn.Linear(((img_size // (2 ** n_down_blocks)) ** 2) * curr_dim, lat_dim, bias=False),
            nn.LeakyReLU(.2, True),
            nn.Linear(lat_dim, lat_dim, bias=False),
            nn.LeakyReLU(.2, True)
        ]

        self.gamma = nn.Linear(lat_dim, lat_dim, bias=False)
        self.beta = nn.Linear(lat_dim, lat_dim, bias=False)

        self.mlp_model = nn.Sequential(*mlp_layers)

    def forward(self, x):
        z_style, z_style_cam_logit = self.shared_style_encoder(x, mask=None)

        out = self.mlp_model(z_style.view(z_style.shape[0], -1))
        gamma, beta = self.gamma(out), self.beta(out)
        return z_style, z_style_cam_logit, (gamma, beta)


class UpSampleBlock(nn.Module):
    """Up-Sampling Module"""

    def __init__(self, conv_dim: int = 64):
        super(UpSampleBlock, self).__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(conv_dim, conv_dim // 2, kernel_size=3, stride=1, padding=0, bias=False)
        # self.norm = AdaILN(conv_dim // 2)
        self.norm = ILN(conv_dim // 2)
        self.act = nn.LeakyReLU(.2, True)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.pad(x)
        x = self.conv(x)
        # x = self.norm(x, gamma, beta)
        x = self.norm(x)
        x = self.act(x)
        return x


class E1(nn.Module):
    def __init__(self, n_feats: int = 32, sep: int = 256,
                 n_blocks: int = 3, n_res_blocks: int = 6):
        super(E1, self).__init__()

        self.sep = sep

        n_f = n_feats

        layers = list()
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, n_f, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(n_feats * 1),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_blocks):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_f * 1, n_f * 2,
                          kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(n_f * 2),
                nn.LeakyReLU(.2, True)
            ]
            n_f *= 2

        out_n_f: int = n_f * 2 - self.sep
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_f, out_n_f, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(out_n_f),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_res_blocks):
            layers += [
                ResidualBlock(out_n_f, out_n_f, use_bias=False)
            ]

        self.gap_fc = nn.Linear(out_n_f, 1, bias=False)
        self.gmp_fc = nn.Linear(out_n_f, 1, bias=False)
        self.conv = nn.Conv2d(out_n_f * 2, out_n_f, kernel_size=1, stride=1, bias=True)
        self.act = nn.LeakyReLU(.2, True)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x_out = self.model(x)

        gap = F.adaptive_avg_pool2d(x_out, 1)
        gap_logit = self.gap_fc(gap.view(x_out.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x_out * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x_out, 1)
        gmp_logit = self.gmp_fc(gmp.view(x_out.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x_out * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.conv(x)
        out = self.act(x)
        return out, cam_logit


class E2(nn.Module):
    def __init__(self, n_feats: int = 32, sep: int = 256,
                 n_blocks: int = 3, n_res_blocks: int = 6):
        super(E2, self).__init__()

        self.sep = sep

        n_f = n_feats

        layers = list()
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, n_f, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(n_feats * 1),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_blocks):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_f * 1, n_f * 2,
                          kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(n_f * 2),
                nn.LeakyReLU(.2, True)
            ]
            n_f *= 2

        out_n_f: int = self.sep
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_f, out_n_f, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(out_n_f),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_res_blocks):
            layers += [
                ResidualBlock(out_n_f, out_n_f, use_bias=False)
            ]

        self.gap_fc = nn.Linear(out_n_f, 1, bias=False)
        self.gmp_fc = nn.Linear(out_n_f, 1, bias=False)
        self.conv = nn.Conv2d(out_n_f * 2, out_n_f, kernel_size=1, stride=1, bias=True)
        self.act = nn.LeakyReLU(.2, True)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x_out = self.model(x)

        gap = F.adaptive_avg_pool2d(x_out, 1)
        gap_logit = self.gap_fc(gap.view(x_out.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x_out * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x_out, 1)
        gmp_logit = self.gmp_fc(gmp.view(x_out.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x_out * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.conv(x)
        out = self.act(x)
        return out, cam_logit


class Decoder(nn.Module):
    def __init__(self, size):
        super(Decoder, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, net):
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net


class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net

