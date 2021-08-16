import ntpath
import os
import itertools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import gather, parallel_apply, replicate
from torchprofile import profile_macs
from thop import profile
from tqdm import tqdm

from data import create_eval_dataloader
from metric import get_fid, get_cityscapes_mIoU
from metric.cityscapes_mIoU import DRNSeg
from metric.inception import InceptionV3
from utils import util
from . import networks
from utils.vgg_feature import VGGFeature
from .base_model import BaseModel
from .modules.loss import GANLoss
from models.modules import pytorch_ssim

class Pix2PixModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert is_train
        parser = super(Pix2PixModel, Pix2PixModel).modify_commandline_options(parser, is_train)
        parser.add_argument('--restore_G_path', type=str, default=None,
                            help='the path to restore the generator')
        parser.add_argument('--restore_D_path', type=str, default=None,
                            help='the path to restore the discriminator')
        parser.add_argument('--recon_loss_type', type=str, default='l1',
                            choices=['l1', 'l2', 'smooth_l1'],
                            help='the type of the reconstruction loss')
        parser.add_argument('--lambda_recon', type=float, default=100,
                            help='weight for reconstruction loss')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight for gan loss')

        parser.add_argument('--real_stat_path', type=str, required=True,
                            help='the path to load the groud-truth images information to compute FID.')

        parser.add_argument('--project', type=str, default=None, help='the project name of this trail')
        parser.add_argument('--name', type=str, default=None, help='the name of this trail')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert opt.isTrain
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_gan', 'G_recon','D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G','D',]
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, opt.dropout_rate, opt.init_type,
                                      opt.init_gain, self.gpu_ids, opt=opt)


        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        # define loss functions
        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        if opt.recon_loss_type == 'l1':
            self.criterionRecon = torch.nn.L1Loss()
        elif opt.recon_loss_type == 'l2':
            self.criterionRecon = torch.nn.MSELoss()
        elif opt.recon_loss_type == 'smooth_l1':
            self.criterionRecon = torch.nn.SmoothL1Loss()
        else:
            raise NotImplementedError('Unknown reconstruction loss type [%s]!' % opt.loss_type)
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(self.device)
        self.inception_model.eval()
        self.vgg = VGGFeature().to(self.device)

        if 'cityscapes' in opt.dataroot:
            self.drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
            util.load_network(self.drn_model, opt.drn_path, verbose=False)
            if len(opt.gpu_ids) > 0:
                self.drn_model.to(self.device)
                self.drn_model = nn.DataParallel(self.drn_model, opt.gpu_ids)
            self.drn_model.eval()

        self.best_fid, self.best_fid_1, self.best_fid_ensemble = 1e9, 1e9, 1e9
        self.best_mIoU, self.best_mIoU_1, self.best_mIoU_ensemble = -1e9, -1e9, -1e9
        self.is_best, self.is_best_1, self.is_best_ensemble = False, False, False
        self.npz = np.load(opt.real_stat_path)

        self.netG.train()
        self.netD.train()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).detach()

        real_AB = torch.cat((self.real_A, self.real_B), 1).detach()
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan

        # Second, G(A) = B
        self.loss_G_recon = self.criterionRecon(self.fake_B, self.real_B) * self.opt.lambda_recon
        # combine loss and calculate gradients

        self.loss_G = self.loss_G_gan + self.loss_G_recon
        self.loss_G.backward()

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def optimize_parameters(self, steps):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

    def evaluate_model(self, step):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG.eval()
        fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
            self.set_input(data_i)
            self.test()
            fakes.append(self.fake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10:
                    input_im = util.tensor2im(self.real_A[j])
                    real_im = util.tensor2im(self.real_B[j])
                    fake_im = util.tensor2im(self.fake_B[j])
                    util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png' % name), create_dir=True)
                    util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    util.save_image(fake_im, os.path.join(save_dir, 'fake', '%s.png' % name), create_dir=True)
                cnt += 1
        fid = get_fid(fakes, self.inception_model, self.npz, device=self.device,
                      batch_size=self.opt.eval_batch_size, tqdm_position=2)
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        ret = {'metric/fid': fid, 'metric/fid-best': self.best_fid}

        if 'cityscapes' in self.opt.dataroot:
            mIoU = get_cityscapes_mIoU(fakes, names, self.drn_model, self.device,
                                       table_path=self.opt.table_path,
                                       data_dir=self.opt.cityscapes_path,
                                       batch_size=self.opt.eval_batch_size,
                                       num_workers=self.opt.num_threads, tqdm_position=2)
            if mIoU > self.best_mIoU:
                self.is_best = True
                self.best_mIoU = mIoU
            ret['metric/mIoU'] = mIoU
            ret['metric/mIoU-best'] = self.best_mIoU
        self.netG.train()
        return ret

    def profile(self, config=None, verbose=True):
        netG = self.netG
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        if config is not None:
            netG.configs = config
        with torch.no_grad():
            macs = profile_macs(netG, (self.real_A[:1],))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params