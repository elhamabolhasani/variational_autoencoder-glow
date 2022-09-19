"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from vae_n_d_n_l import VAE
import util
from util import *
from models import Glow
from torch.distributions import LowRankMultivariateNormal
from torch.distributions import Independent, Normal
from tqdm import tqdm


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since Glow expects inputs in (0, 1)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model vae
    vae_net = VAE('cifar')
    vae_net.init_model()
    vae_net.load_state_dict(torch.load(args.vae_model_path))
    vae_net.eval()  # test remove

    # Model glow
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global global_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, vae_net, trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm)
        test(epoch, net, vae_net, testloader, device, loss_fn, args.num_samples)


@torch.enable_grad()
def train(epoch, net, vae_net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()

            # vae model both lr
            # mu_d, logvar_d, u_d, mu, logvar, u = vae_net(x)

            # vae model both n
            mu_d, logvar_d, mu, logvar= vae_net(x)

            # glow model
            z, sldj = net(x, reverse=False)
            # loss = loss_fn(z, sldj, mu_d, logvar_d,u_d)
            loss = loss_fn(z, sldj, mu_d, logvar_d)

            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)


@torch.no_grad()
def sample(net, vae_net, batch_size, device):
    # assume latent features space ~ N(0, 1)
    z = torch.randn(batch_size, vae_net.n_latent_features).to(device)
    z = vae_net.fc4(z)
    z = vae_net.decoder(z)
    z = z.view(-1, vae_net.n_neurons_last_decoder_layer)

    """ both normal """
    mu_d, logvar_d = vae_net.decoder_bottleneck(z)
    dist = Independent(Normal(loc=mu_d, scale=torch.exp(logvar_d)), 1)

    """ both low rank """
    # mu_d, logvar_d, u_d = vae_net.decoder_bottleneck(z)
    #
    # sigma = torch.exp(logvar_d) + 0.001
    # dist = LowRankMultivariateNormal(mu_d, u_d.view(-1, u_d.shape[1], 1), sigma)

    #sample from model
    z = dist.sample()
    z = z.view(-1, 3, 32, 32)

    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)
    return x


@torch.no_grad()
def test(epoch, net, vae_net, testloader, device, loss_fn, num_samples):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            # vae model
            # mu_d, logvar_d, u_d, mu, logvar, u = vae_net(x)
            mu_d, logvar_d, mu, logvar = vae_net(x)

            z, sldj = net(x, reverse=False)
            # loss = loss_fn(z, sldj, mu_d, logvar_d,u_d)
            loss = loss_fn(z, sldj, mu_d, logvar_d)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    print('best_loss ', best_loss)
    print('loss_meter.avg  ', loss_meter.avg)
    if loss_meter.avg < best_loss:
        best_loss = loss_meter.avg

    print('Saving...')
    state = {
        'net': net.state_dict(),
        'test_loss': loss_meter.avg,
        'epoch': epoch,
    }
    os.makedirs('ckpts', exist_ok=True)
    torch.save(state, 'ckpts/glow_' + str(epoch) + 'vae(n_d_n_l_256).pth.tar')

    # Save samples and data
    images = sample(net, vae_net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')  # TODO 1e-3 -> 1e-
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--vae_model_path', default="vae_ckpts/vae_n_decoder_n_latent_dim=256_cifar_model_1001.pt",
                                type=str, help='Number of steps for lr warm-up') # TODO change here
    parser.add_argument('--vae_optim_path', default="vae_n_decoder_n_latent_dimcifar_optim_1001.pt",
                                type=str, help='Number of steps for lr warm-up') # TODO change here

    best_loss = 0
    global_step = 0
    args = parser.parse_args(args=[])
    main(args)


# "E:\variational autoencoder and normal flows\venv\Scripts\python.exe" "E:/variational autoencoder and normal flows/glow_vae/glow-master/glow_model.py"
# Files already downloaded and verified
# Files already downloaded and verified
# 3072
# Building model..
#
# Epoch: 0
#   0%|          | 0/50000 [00:00<?, ?it/s]E:\variational autoencoder and normal flows\venv\lib\site-packages\torch\optim\lr_scheduler.py:154: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
#   warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
# 100%|██████████| 50000/50000 [17:19<00:00, 48.12it/s, bpd=15.9, lr=0.0001, nll=3.39e+4]
# 100%|██████████| 10000/10000 [00:54<00:00, 184.69it/s, bpd=-2.47, nll=-5.26e+3]
# best_loss  0
# loss_meter.avg   -5258.58814609375
# Saving...
#
# Epoch: 1
# 100%|██████████| 50000/50000 [16:11<00:00, 51.44it/s, bpd=-1.22, lr=0.0002, nll=-2.6e+3]
# 100%|██████████| 10000/10000 [01:01<00:00, 162.63it/s, bpd=4.06e+9, nll=8.65e+12]
# best_loss  -5258.58814609375
# loss_meter.avg   8649321134880.594
# Saving...
#
# Epoch: 2
# 100%|██████████| 50000/50000 [16:40<00:00, 50.00it/s, bpd=5.02e+21, lr=0.0003, nll=1.07e+25]
# 100%|██████████| 10000/10000 [01:02<00:00, 160.85it/s, bpd=5.91e+24, nll=1.26e+28]
# best_loss  -5258.58814609375
# loss_meter.avg   1.2587809256451093e+28
# Saving...
#
# Epoch: 3
#  60%|█████▉    | 29856/50000 [10:23<07:00, 47.86it/s, bpd=5.43e+21, lr=0.00036, nll=1.16e+25]
# Traceback (most recent call last):
#   File "E:/variational autoencoder and normal flows/glow_vae/glow-master/glow_model.py", line 224, in <module>
#     main(args)
#   File "E:/variational autoencoder and normal flows/glow_vae/glow-master/glow_model.py", line 87, in main
#     loss_fn, args.max_grad_norm)
#   File "E:\variational autoencoder and normal flows\venv\lib\site-packages\torch\autograd\grad_mode.py", line 28, in decorate_context
#     return func(*args, **kwargs)
#   File "E:/variational autoencoder and normal flows/glow_vae/glow-master/glow_model.py", line 111, in train
#     loss = loss_fn(z, sldj, mu_d, logvar_d)
#   File "E:\variational autoencoder and normal flows\venv\lib\site-packages\torch\nn\modules\module.py", line 1102, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "E:\variational autoencoder and normal flows\glow_vae\glow-master\util\optim_util.py", line 61, in forward
#     d = dist.log_prob(z.view(-1, z.shape[2] * z.shape[2] * self.color_channels))
#   File "E:\variational autoencoder and normal flows\venv\lib\site-packages\torch\distributions\independent.py", line 91, in log_prob
#     log_prob = self.base_dist.log_prob(value)
#   File "E:\variational autoencoder and normal flows\venv\lib\site-packages\torch\distributions\normal.py", line 73, in log_prob
#     self._validate_sample(value)
#   File "E:\variational autoencoder and normal flows\venv\lib\site-packages\torch\distributions\distribution.py", line 289, in _validate_sample
#     "Expected value argument "
# ValueError: Expected value argument (Tensor of shape (32, 3072)) to be within the support (Real()) of the distribution Normal(loc: torch.Size([32, 3072]), scale: torch.Size([32, 3072])), but found invalid values:
# tensor([[nan, nan, nan,  ..., nan, nan, nan],
#         [nan, nan, nan,  ..., nan, nan, nan],
#         [nan, nan, nan,  ..., nan, nan, nan],
#         ...,
#         [nan, nan, nan,  ..., nan, nan, nan],
#         [nan, nan, nan,  ..., nan, nan, nan],
#         [nan, nan, nan,  ..., nan, nan, nan]], device='cuda:0',
#        grad_fn=<ViewBackward0>)
#
# Process finished with exit code 1
