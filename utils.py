import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def save_images(args, e1, e2, decoder, iters):
    test_domain_a, test_domain_b = get_test_images(args)

    exps = []
    for i in range(args.num_display):
        with torch.no_grad():
            if i == 0:
                filler = test_domain_b[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))
            exps.append(test_domain_b[i].unsqueeze(0))

    for i in range(args.num_display):
        exps.append(test_domain_a[i].unsqueeze(0))
        separate_a, _ = e2(test_domain_a[i].unsqueeze(0))
        for j in range(args.num_display):
            with torch.no_grad():
                common_b = e1(test_domain_b[j].unsqueeze(0))

                ba_encoding = torch.cat([common_b, separate_a], dim=1)
                ba_decoding = decoder(ba_encoding)
                exps.append(ba_decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/experiments_%06d.png' % (args.out, iters),
                      normalize=True, nrow=args.num_display + 1)


def interpolate(args, e1, e2, decoder):
    test_domain_a, test_domain_b = get_test_images(args)

    exps = []
    _inter_size = 5
    with torch.no_grad():
        for i in range(5):
            b_img = test_domain_b[i].unsqueeze(0)
            common_b, _ = e1(b_img)
            for j in range(args.num_display):
                with torch.no_grad():
                    exps.append(test_domain_a[j].unsqueeze(0))

                    separate_a_1, _ = e2(test_domain_a[j].unsqueeze(0))
                    separate_a_2, _ = e2(test_domain_a[j].unsqueeze(0))
                    for k in range(_inter_size + 1):
                        cur_sep = float(j) / _inter_size * separate_a_2 + (1 - (float(k) / _inter_size)) * separate_a_1
                        a_encoding = torch.cat([common_b, cur_sep], dim=1)
                        a_decoding = decoder(a_encoding)
                        exps.append(a_decoding)
                    exps.append(test_domain_a[i].unsqueeze(0))

            exps = torch.cat(exps, 0)
            vutils.save_image(exps,
                              '%s/interpolation.png' % args.save,
                              normalize=True, nrow=_inter_size + 3)


def get_test_images(args):
    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domain_a_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=comp_transform)
    domain_b_test = CustomDataset(os.path.join(args.root, 'testB.txt'), transform=comp_transform)

    domain_a_test_loader = torch.utils.data.DataLoader(domain_a_test, batch_size=64,
                                                       shuffle=False, num_workers=6)
    domain_b_test_loader = torch.utils.data.DataLoader(domain_b_test, batch_size=64,
                                                       shuffle=False, num_workers=6)

    for domain_a_img in domain_a_test_loader:
        domain_a_img = Variable(domain_a_img)
        if torch.cuda.is_available():
            domain_a_img = domain_a_img.cuda()
        domain_a_img = domain_a_img.view((-1, 3, args.resize, args.resize))
        domain_a_img = domain_a_img[:]
        break

    for domain_b_img in domain_b_test_loader:
        domain_b_img = Variable(domain_b_img)
        if torch.cuda.is_available():
            domain_b_img = domain_b_img.cuda()
        domain_b_img = domain_b_img.view((-1, 3, args.resize, args.resize))
        domain_b_img = domain_b_img[:]
        break

    return domain_a_img, domain_b_img


def save_model(out_file, e1, e2, decoder, ae_opt, disc, disc_opt, iters):
    state = {
        'e1': e1.state_dict(),
        'e2': e2.state_dict(),
        'decoder': decoder.state_dict(),
        'ae_opt': ae_opt.state_dict(),
        'disc': disc.state_dict(),
        'disc_opt': disc_opt.state_dict(),
        'iters': iters
    }
    torch.save(state, out_file)


def load_model(load_path: str, e1, e2, decoder, ae_opt, disc, disc_opt):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    decoder.load_state_dict(state['decoder'])
    ae_opt.load_state_dict(state['ae_opt'])
    disc.load_state_dict(state['disc'])
    disc_opt.load_state_dict(state['disc_opt'])
    return state['iters']


def load_model_for_eval(load_path: str, e1, e2, decoder):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    decoder.load_state_dict(state['decoder'])
    return state['iters']


class CustomDataset(data.Dataset):
    def __init__(self, path: str, transform=None, return_paths: bool = False):
        super(CustomDataset, self).__init__()

        with open(path) as f:
            images = [s.replace('\n', '') for s in f.readlines()]

        self.images = images
        self.transform = transform
        self.return_paths = return_paths

    @staticmethod
    def loader(path: str):
        # return cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1][20:-20, ...]
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        path = self.images[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.images)
