import itertools
import os

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel

from torchprofile import profile_macs
import models.modules.loss
from utils.image_pool import ImagePool
from data import create_eval_dataloader
from metric import create_metric_models
from models import networks
from models.base_model import BaseModel
from utils import util
from models.modules.loss import GANLoss
from data import create_dataloader
import math


class BaseCycleganBestDistiller(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(BaseCycleganBestDistiller, BaseCycleganBestDistiller).modify_commandline_options(parser, is_train)
        parser.add_argument('--restore_teacher_G_A_path', type=str, default=None,
                            help='the path to restore the generator G_A')
        parser.add_argument('--restore_teacher_D_A_path', type=str, default=None,
                            help='the path to restore the discriminator D_A')
        parser.add_argument('--restore_teacher_G_B_path', type=str, default=None,
                            help='the path to restore the generator G_B')
        parser.add_argument('--restore_teacher_D_B_path', type=str, default=None,
                            help='the path to restore the discriminator D_B')

        parser.add_argument('--lambda_A', type=float, default=10.0,
                            help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0,
                            help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='use identity mapping. '
                                 'Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. '
                                 'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        parser.add_argument('--real_stat_A_path', type=str, required=True,
                            help='the path to load the ground-truth A images information to compute FID.')
        parser.add_argument('--real_stat_B_path', type=str, required=True,
                            help='the path to load the ground-truth B images information to compute FID.')

        parser.add_argument('--teacher_netG', type=str, default='mobile_resnet_9blocks',
                            help='specify teacher generator architecture')
        parser.add_argument('--student_netG', type=str, default='mobile_resnet_9blocks',
                            help='specify student generator architecture')
        parser.add_argument('--teacher_ngf', type=int, default=64,
                            help='the base number of filters of the teacher generator')
        parser.add_argument('--student_ngf', type=int, default=16,
                            help='the base number of filters of the student generator')
        parser.add_argument('--restore_student_G_path', type=str, default=None,
                            help='the path to restore the student generator')
        parser.add_argument('--restore_A_path', type=str, default=None,
                            help='the path to restore the adaptors for distillation')
        parser.add_argument('--restore_O_path', type=str, default=None,
                            help='the path to restore the optimizer')
        parser.add_argument('--recon_loss_type', type=str, default='l1',
                            choices=['l1', 'l2', 'smooth_l1', 'vgg'],
                            help='the type of the reconstruction loss')

        parser.add_argument('--lambda_CD', type=float, default=0,
                            help='weights for the intermediate activation distillation loss')
        parser.add_argument('--lambda_recon', type=float, default=100,
                            help='weights for the reconstruction loss.')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight for gan loss')

        parser.add_argument('--teacher_dropout_rate', type=float, default=0)
        parser.add_argument('--student_dropout_rate', type=float, default=0)

        parser.add_argument('--project', type=str, default=None, help='the project name of this trail')
        parser.add_argument('--name', type=str, default=None, help='the name of this trail')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.set_defaults(norm='instance', dataset_mode='unaligned',
                            batch_size=1, ndf=64, gan_mode='lsgan',
                            nepochs=100, nepochs_decay=100, save_epoch_freq=20)
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        assert opt.dataset_mode == 'unaligned'
        super(BaseCycleganBestDistiller, self).__init__(opt)
        self.loss_names = ['D_A', 'G_A', 'G_cycle_A', 'G_idt_A',
                           'D_B', 'G_B', 'G_cycle_B', 'G_idt_B',
                           'G_SSIM', 'G_style', 'G_feature', 'G_tv', 'G_CD']
        self.optimizers = []
        self.image_paths = []
        self.visual_names_A = ['real_A', 'fake_B', 'rec_A']
        self.visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.opt.lambda_identity > 0.0:
            self.visual_names_A.append('idt_B')
            self.visual_names_B.append('idt_A')
        self.model_names = ['netG_student', 'netG_teacher_A', 'netG_teacher_B', 'netD_teacher_A', 'netD_teacher_B']

        self.netG_teacher_A = networks.define_G(opt.input_nc, opt.output_nc, opt.teacher_ngf,
                                                opt.teacher_netG, opt.norm, opt.teacher_dropout_rate,
                                                opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)
        self.netG_teacher_B = networks.define_G(opt.input_nc, opt.output_nc, opt.teacher_ngf,
                                                opt.teacher_netG, opt.norm, opt.teacher_dropout_rate,
                                                opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        self.netG_student = networks.define_G(opt.input_nc, opt.output_nc, opt.student_ngf,
                                              opt.student_netG, opt.norm, opt.student_dropout_rate,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        self.netD_teacher_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm,
                                                opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_teacher_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm,
                                                opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_teacher_A.train()
        self.netG_teacher_B.train()
        self.netG_student.train()
        self.netD_teacher_A.train()
        self.netD_teacher_B.train()

        if self.opt.lambda_CD:
            self.mapping_layers = {'mobile_resnet_9blocks':['model.9',  # 4 * ngf
                                                            'model.12',
                                                            'model.15',
                                                            'model.18']}
        self.netAs = []
        self.Tacts, self.Sacts = {}, {}
        G_params = [self.netG_student.parameters()]
        if self.opt.lambda_CD:
            for i, n in enumerate(self.mapping_layers[self.opt.teacher_netG]):
                ft, fs = self.opt.teacher_ngf, self.opt.student_ngf
                if 'resnet' in self.opt.teacher_netG:
                    netA = self.build_feature_connector(4 * ft, 4 * fs)
                elif i == 0:
                    netA = self.build_feature_connector(2 * ft, 2 * fs)
                elif i == 1:
                    netA = self.build_feature_connector(8 * ft, 8 * fs)
                elif i == 2:
                    netA = self.build_feature_connector(16 * ft, 16 * fs)
                else:
                    netA = self.build_feature_connector(4 * ft, 4 * fs)
                networks.init_net(netA)
                G_params.append(netA.parameters())
                self.netAs.append(netA)

        self.criterionGAN = models.modules.loss.GANLoss(opt.gan_mode).to(self.device)
        self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(opt.pool_size)
        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        self.optimizer_G_student = torch.optim.Adam(itertools.chain(*G_params), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G_teacher = torch.optim.Adam(itertools.chain(self.netG_teacher_A.parameters(),
                                                                    self.netG_teacher_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_teacher = torch.optim.Adam(itertools.chain(self.netD_teacher_A.parameters(),
                                                                    self.netD_teacher_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G_student)
        self.optimizers.append(self.optimizer_D_teacher)
        self.optimizers.append(self.optimizer_G_teacher)

        self.eval_dataloader_AtoB = create_eval_dataloader(self.opt, direction='AtoB')
        self.eval_dataloader_BtoA = create_eval_dataloader(self.opt, direction='BtoA')

        self.inception_model, self.drn_model = create_metric_models(opt, device=self.device)
        self.npz_A = np.load(opt.real_stat_A_path)
        self.npz_B = np.load(opt.real_stat_B_path)
        self.is_best = False
        self.student_steps = 0
        self.student_epoch = 0
        self.student_dataloader = create_dataloader(self.opt)

    def setup(self, opt, verbose=True):
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.load_networks(verbose)
        if verbose:
            self.print_networks()
        if self.opt.lambda_CD > 0:
            def get_activation(mem, name):
                def get_output_hook(module, input, output):
                    mem[name + str(output.device)] = output

                return get_output_hook

            def add_hook(net, mem, mapping_layers):
                for n, m in net.named_modules():
                    if n in mapping_layers:
                        m.register_forward_hook(get_activation(mem, n))

            add_hook(self.netG_teacher_A, self.Tacts, self.mapping_layers[self.opt.teacher_netG])
            add_hook(self.netG_student, self.Sacts, self.mapping_layers[self.opt.teacher_netG])

    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel),
             nn.ReLU(inplace=True)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        raise NotImplementedError

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_teacher_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_teacher_B, self.real_A, fake_A)

    def backward_G(self):
        raise NotImplementedError

    def optimize_parameters(self, steps):
        raise NotImplementedError

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if hasattr(self, name):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                with open(os.path.join(self.opt.log_dir, name + '.txt'), 'w') as f:
                    f.write(str(net) + '\n')
                    f.write('[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, verbose=True):
        if self.opt.restore_teacher_G_A_path is not None:
            util.load_network(self.netG_teacher_A, self.opt.restore_teacher_G_A_path, verbose)
        if self.opt.restore_teacher_G_B_path is not None:
            util.load_network(self.netG_teacher_B, self.opt.restore_teacher_G_B_path, verbose)
        if self.opt.restore_student_G_path is not None:
            util.load_network(self.netG_student, self.opt.restore_student_G_path, verbose)
        if self.opt.restore_teacher_D_A_path is not None:
            util.load_network(self.netD_teacher_A, self.opt.restore_D_A_path, verbose)
        if self.opt.restore_teacher_D_B_path is not None:
            util.load_network(self.netD_teacher_B, self.opt.restore_D_B_path, verbose)
        if self.opt.restore_A_path is not None:
            for i, netA in enumerate(self.netAs):
                path = '%s-%d.pth' % (self.opt.restore_A_path, i)
                util.load_network(netA, path, verbose)
        if self.opt.restore_O_path is not None:
            for i, optimizer in enumerate(self.optimizers):
                path = '%s-%d.pth' % (self.opt.restore_O_path, i)
                util.load_optimizer(optimizer, path, verbose)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opt.lr

    def save_net(self, net, save_path):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            if isinstance(net, DataParallel):
                torch.save(net.module.cpu().state_dict(), save_path)
            else:
                torch.save(net.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def save_networks(self, epoch):
        teacher_save_dir = os.path.join(self.save_dir, 'teacher')
        os.makedirs(teacher_save_dir, exist_ok=True)

        save_filename = '%s_net_%s_teacher_A.pth' % (epoch, 'G')
        save_path = os.path.join(teacher_save_dir, save_filename)
        net = getattr(self, 'net%s_teacher_A' % 'G')
        self.save_net(net, save_path)

        save_filename = '%s_net_%s_teacher_B.pth' % (epoch, 'G')
        save_path = os.path.join(teacher_save_dir, save_filename)
        net = getattr(self, 'net%s_teacher_B' % 'G')
        self.save_net(net, save_path)

        save_filename = '%s_net_%s_teacher_A.pth' % (epoch, 'D')
        save_path = os.path.join(teacher_save_dir, save_filename)
        net = getattr(self, 'net%s_teacher_A' % 'D')
        self.save_net(net, save_path)

        save_filename = '%s_net_%s_teacher_B.pth' % (epoch, 'D')
        save_path = os.path.join(teacher_save_dir, save_filename)
        net = getattr(self, 'net%s_teacher_B' % 'D')
        self.save_net(net, save_path)

        for i, optimizer in enumerate(self.optimizers):
            save_filename = '%s_optim-%d.pth' % (epoch, i)
            save_path = os.path.join(teacher_save_dir, save_filename)
            torch.save(optimizer.state_dict(), save_path)

    def save_student_networks(self, epoch):
        student_save_dir = os.path.join(self.save_dir, 'student')
        os.makedirs(student_save_dir, exist_ok=True)

        save_filename = '%s_net_%s.pth' % (epoch, 'G')
        save_path = os.path.join(student_save_dir, save_filename)
        net = getattr(self, 'net%s_student' % 'G')
        self.save_net(net, save_path)

        for i, net in enumerate(self.netAs):
            save_filename = '%s_net_%s-%d.pth' % (epoch, 'A', i)
            save_path = os.path.join(student_save_dir, save_filename)
            self.save_net(net, save_path)


    def evaluate_model(self, step):
        raise NotImplementedError

    def test(self):
        with torch.no_grad():
            self.forward()

    def profile(self, config=None, verbose=True):
        for name in self.model_names:
            if hasattr(self,name) and 'D' not in name:
                netG = getattr(self,name)
                if isinstance(netG, nn.DataParallel):
                    netG = netG.module
                if config is not None:
                    netG.configs = config
                with torch.no_grad():
                    macs = profile_macs(netG, (self.real_A[:1],))
                    # flops, params = profile(netG, inputs=(self.real_A[:1],))
                params = 0
                for p in netG.parameters():
                    params += p.numel()
                if verbose:
                    print('%s : MACs: %.3fG\tParams: %.3fM' % (name, macs / 1e9, params / 1e6), flush=True)
        return None