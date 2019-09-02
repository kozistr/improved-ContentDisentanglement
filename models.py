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


class UpSampleBlock(nn.Module):
    """Up-Sampling Module"""
    def __init__(self, conv_dim: int = 64):
        super(UpSampleBlock, self).__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(conv_dim, conv_dim // 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm = ILN(conv_dim // 2)
        self.act = nn.LeakyReLU(.2, True)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class E1(nn.Module):
    def __init__(self, size: int = 128, n_feats: int = 32, sep: int = 256,
                 n_blocks: int = 3, n_res_blocks: int = 6):
        super(E1, self).__init__()

        self.size = size
        self.sep = sep

        n_f = n_feats

        layers = list()
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, n_f, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(n_f),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_blocks):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_f, n_f * 2,
                          kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(n_f * 2),
                nn.LeakyReLU(.2, True)
            ]
            n_f *= 2

        out_n_f: int = n_f * 2 - self.sep
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_f, out_n_f, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(out_n_f),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_res_blocks):
            layers += [ResidualBlock(out_n_f, out_n_f, use_bias=False)]

        mlp_layers = [
            nn.Linear(out_n_f * ((self.size // (2 ** (n_blocks + 1))) ** 2), n_f * 2, bias=False),
            nn.LeakyReLU(.2, True),
            nn.Linear(n_f * 2, n_f * 2, bias=False),
            nn.LeakyReLU(.2, True)
        ]

        self.gamma = nn.Linear(n_f * 2, n_f * 2, bias=False)
        self.beta = nn.Linear(n_f * 2, n_f * 2, bias=False)

        self.mlp_model = nn.Sequential(*mlp_layers)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x_out = self.model(x)

        out = self.mlp_model(x_out.view(x_out.shape[0], -1))
        gamma, beta = self.gamma(out), self.beta(out)
        return x_out, (gamma, beta)


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
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_f, out_n_f, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(out_n_f),
            nn.LeakyReLU(.2, True)
        ]

        for i in range(n_res_blocks):
            layers += [
                ResidualBlock(out_n_f, out_n_f, use_bias=False)
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_feats: int = 512,
                 n_blocks: int = 4, n_res_blocks: int = 6):
        super(Decoder, self).__init__()

        self.n_res_blocks = n_res_blocks

        n_f: int = n_feats

        res_layers = [
            ResidualAdaLINBlock(n_f, n_f, use_bias=False)
            for _ in range(n_res_blocks)
        ]

        layers = list()
        for i in range(n_blocks):
            layers += [UpSampleBlock(n_f)]
            n_f //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_f, 3, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        ]

        self.res_model = nn.Sequential(*res_layers)
        self.model = nn.Sequential(*layers)

    def forward(self, x, gamma, beta):
        for i in range(self.n_res_blocks):
            x = self.res_model[i](x, gamma, beta)
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    """Discriminator Network w/ PatchGAN"""

    def __init__(self, in_ch: int = 3, conv_dim: int = 64, n_down_blocks: int = 5):
        super(Discriminator, self).__init__()

        layers = list()
        layers += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, conv_dim, kernel_size=4, stride=2, padding=0, bias=True)
            ),
            nn.LeakyReLU(.2, True)
        ]

        curr_dim: int = conv_dim
        for i in range(1, n_down_blocks - 2):
            layers += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=0, bias=True)
                ),
                nn.LeakyReLU(.2, True)
            ]
            curr_dim *= 2

        layers += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=0, bias=True)
            ),
            nn.LeakyReLU(.2, True)
        ]
        curr_dim *= 2

        self.gap_fc = nn.utils.spectral_norm(nn.Linear(curr_dim, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(curr_dim, 1, bias=False))
        self.conv1x1 = nn.Conv2d(curr_dim * 2, curr_dim, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)
        x = torch.cat([gap, gmp], dim=1)

        x = self.conv1x1(x)
        x = self.leaky_relu(x)

        x = self.pad(x)
        x = self.conv(x)

        return x, cam_logit


class Disc(nn.Module):
    def __init__(self, n_feat: int = 512, sep: int = 256, size: int = 128, n_blocks: int = 4):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size
        self.n_feat = n_feat
        self.n_blocks = n_blocks

        self.n_res = (self.size // (2 ** self.n_blocks)) ** 2

        self.classify = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear((self.n_feat - self.sep) * self.n_res, self.n_feat)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Linear(self.n_feat, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, (self.n_feat - self.sep) * self.n_res)
        x = self.classify(x).view(-1)
        return x


class RhoClipper:
    def __init__(self, clip_min: float = 0., clip_max: float = 1.):
        self.clip_min = clip_min
        self.clip_max = clip_max
        assert clip_min < clip_max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
