import argparse
import os

import torch
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import E1
from models import E2
from models import Disc
from models import Decoder
from models import RhoClipper
from utils import CustomDataset
from utils import save_images, save_model, load_model


def train(config):
    if not os.path.exists(config.out):
        os.makedirs(config.out)

    comp_transform = transforms.Compose([
        transforms.CenterCrop(config.crop),
        transforms.Resize(config.resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domain_a_train = CustomDataset(os.path.join(config.root, 'trainA.txt'), transform=comp_transform)
    domain_b_train = CustomDataset(os.path.join(config.root, 'trainB.txt'), transform=comp_transform)

    a_label = torch.full((config.bs,), 1)
    b_label = torch.full((config.bs,), 0)
    b_separate = torch.full((config.bs,
                             config.sep,
                             config.resize // (2 ** (config.n_blocks + 1)),
                             config.resize // (2 ** (config.n_blocks + 1))), 0)

    # build networks
    e1 = E1(sep=config.sep, size=config.resize)
    e2 = E2(n_feats=512, sep=config.sep)
    decoder = Decoder(n_feats=512)
    disc = Disc(size=config.resize, sep=config.sep)
    rho_clipper = RhoClipper(0., 1.)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        decoder = decoder.cuda()
        disc = disc.cuda()

        a_label = a_label.cuda()
        b_label = b_label.cuda()
        b_separate = b_separate.cuda()

        mse = mse.cuda()
        bce = bce.cuda()

    ae_params = list(e1.parameters()) + list(e2.parameters()) + list(decoder.parameters())
    ae_optimizer = optim.Adam(ae_params, lr=config.lr,
                              betas=(config.beta1, config.beta2), eps=config.eps)

    disc_params = disc.parameters()
    disc_optimizer = optim.Adam(disc_params, lr=config.d_lr,
                                betas=(config.beta1, config.beta2), eps=config.eps)

    _iter: int = 0
    if config.load != '':
        save_file = os.path.join(config.load, 'checkpoint')
        _iter = load_model(save_file, e1, e2, decoder, ae_optimizer, disc, disc_optimizer)

    e1 = e1.train()
    e2 = e2.train()
    decoder = decoder.train()
    disc = disc.train()

    print('[*] Started training...')
    while True:
        domain_a_loader = torch.utils.data.DataLoader(domain_a_train, batch_size=config.bs,
                                                      shuffle=True, num_workers=config.n_threads)
        domain_b_loader = torch.utils.data.DataLoader(domain_b_train, batch_size=config.bs,
                                                      shuffle=True, num_workers=config.n_threads)
        if _iter >= config.iters:
            break

        for domain_a_img, domain_b_img in zip(domain_a_loader, domain_b_loader):
            if domain_a_img.size(0) != config.bs or domain_b_img.size(0) != config.bs:
                break

            domain_a_img = Variable(domain_a_img)
            domain_b_img = Variable(domain_b_img)

            if torch.cuda.is_available():
                domain_a_img = domain_a_img.cuda()
                domain_b_img = domain_b_img.cuda()

            domain_a_img = domain_a_img.view((-1, 3, config.resize, config.resize))
            domain_b_img = domain_b_img.view((-1, 3, config.resize, config.resize))

            ae_optimizer.zero_grad()

            a_common = e1(domain_a_img)
            a_separate = e2(domain_a_img)
            a_encoding = torch.cat([a_common, a_separate], dim=1)

            b_common = e1(domain_b_img)
            b_encoding = torch.cat([b_common, b_separate], dim=1)

            a_decoding = decoder(a_encoding)
            b_decoding = decoder(b_encoding)

            g_loss = mse(a_decoding, domain_a_img) + mse(b_decoding, domain_b_img)

            preds_a = disc(a_common)
            preds_b = disc(b_common)
            g_loss += config.adv_weight * (bce(preds_a, b_label) + bce(preds_b, b_label))

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, 5.)
            ae_optimizer.step()

            disc_optimizer.zero_grad()

            a_common = e1(domain_a_img)
            b_common = e1(domain_b_img)

            disc_a = disc(a_common)
            disc_b = disc(b_common)

            d_loss = bce(disc_a, a_label) + bce(disc_b, b_label)

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc_params, 5.)
            disc_optimizer.step()

            decoder.apply(rho_clipper)

            if _iter % config.progress_iter == 0:
                print('[*] [%07d/%07d] d_loss : %.4f, g_loss : %.4f' %
                      (_iter, config.iters, d_loss, g_loss))

            if _iter % config.display_iter == 0:
                e1 = e1.eval()
                e2 = e2.eval()
                decoder = decoder.eval()

                save_images(config, e1, e2, decoder, _iter)

                e1 = e1.train()
                e2 = e2.train()
                decoder = decoder.train()

            if _iter % config.save_iter == 0:
                save_file = os.path.join(config.out, 'checkpoint')
                save_model(save_file, e1, e2, decoder, ae_optimizer, disc, disc_optimizer, _iter)

            _iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='./out')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.999)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--sep', type=int, default=128)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--n_res_blocks', type=int, default=3)
    parser.add_argument('--adv_weight', type=float, default=1e-3)
    parser.add_argument('--d_lr', type=float, default=4e-4)
    parser.add_argument('--progress_iter', type=int, default=100)
    parser.add_argument('--display_iter', type=int, default=5000)
    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--load', default='')
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--num_display', type=int, default=12)

    args = parser.parse_args()

    train(args)
