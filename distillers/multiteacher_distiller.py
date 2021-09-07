import ntpath
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import gather, parallel_apply, replicate
from tqdm import tqdm

from metric import get_fid, get_cityscapes_mIoU
from utils import util
from utils.vgg_feature import VGGFeature
from .base_multiteacher_distiller import BaseMultiTeacherDistiller
from models.modules import pytorch_ssim
from models.modules.discriminators import FLAGS


class MultiTeacherDistiller(BaseMultiTeacherDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(MultiTeacherDistiller, MultiTeacherDistiller).modify_commandline_options(parser, is_train)
        parser.add_argument('--AGD_weights', type=str, default='1e1, 1e4, 1e1, 1e-5', help='weights for losses in AGD mode')
        parser.add_argument('--n_dis', type=int, default=1, help='iter time for student before update teacher')
        parser.set_defaults(norm='instance', dataset_mode='aligned')

        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(MultiTeacherDistiller, self).__init__(opt)
        self.best_fid_teachers, self.best_fid_student = [1e9 for _ in range(self.opt.num_teacher)],  1e9
        self.best_mIoU_teachers, self.best_mIoU_student = [-1e9 for _ in range(self.opt.num_teacher)], -1e9
        self.fids_teacher, self.fids_student, self.mIoUs_teacher, self.mIoUs_student = [], [], [], []
        self.npz = np.load(opt.real_stat_path)
        # weights for AGD mood
        loss_weight = [float(char) for char in opt.AGD_weights.split(',')]
        self.lambda_SSIM = loss_weight[0]
        self.lambda_style = loss_weight[1]
        self.lambda_feature = loss_weight[2]
        self.lambda_tv = loss_weight[3]
        self.vgg = VGGFeature().to(self.device)

    def forward(self):
        self.Tfake_B_w = self.netG_teacher_w(self.real_A)
        self.Tfake_B_d = self.netG_teacher_d(self.real_A)
        self.Tfake_Bs = [self.Tfake_B_w.detach(), self.Tfake_B_d.detach()]
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_CD_loss(self):
        losses = []
        mapping_layers = self.mapping_layers[self.opt.teacher_netG_w]
        for i, netA in enumerate(self.netAs):
            n = mapping_layers[i]
            netA_replicas = replicate(netA.cuda(), self.gpu_ids)
            Sacts = parallel_apply(netA_replicas,
                                       tuple([self.Sacts[key] for key in sorted(self.Sacts.keys()) if n in key]))
            Tacts = [self.Tacts[key] for key in sorted(self.Tacts.keys()) if n in key]
            for Sact, Tact in zip(Sacts, Tacts):
                source, target = Sact, Tact.detach()
                source = source.mean(dim=(2, 3), keepdim=False)
                target = target.mean(dim=(2, 3), keepdim=False)
                loss = torch.mean(torch.pow(source - target, 2))
                losses.append(loss)
        return sum(losses)

    def backward_G_teacher(self):

        fake_AB_w = torch.cat((self.real_A, self.Tfake_B_w), 1)
        FLAGS.teacher_ids = 1
        pred_fake_w = self.netD_teacher(fake_AB_w)
        self.loss_G_gan_w = self.criterionGAN(pred_fake_w, True, for_discriminator=False) * self.opt.lambda_gan
        # Second, G(A) = B
        self.loss_G_recon_w = self.criterionRecon(self.Tfake_B_w, self.real_B) * self.opt.lambda_recon
        # combine loss and calculate gradients
        self.loss_G_w = self.loss_G_gan_w + self.loss_G_recon_w

        fake_AB_d = torch.cat((self.real_A, self.Tfake_B_d), 1)
        FLAGS.teacher_ids = 2
        pred_fake_d = self.netD_teacher(fake_AB_d)
        self.loss_G_gan_d = self.criterionGAN(pred_fake_d, True, for_discriminator=False) * self.opt.lambda_gan
        self.loss_G_recon_d = self.criterionRecon(self.Tfake_B_d, self.real_B) * self.opt.lambda_recon
        self.loss_G_d = self.loss_G_gan_d + self.loss_G_recon_d

        self.loss_G_d.backward()
        self.loss_G_w.backward()


    def backward_G_student(self):
        self.loss_G_student = 0
        for i, teacher_image in enumerate(self.Tfake_Bs):
            ssim_loss = pytorch_ssim.SSIM()
            self.loss_G_SSIM = (1 - ssim_loss(self.Sfake_B, teacher_image)) * self.lambda_SSIM
            Tfeatures = self.vgg(teacher_image)
            Sfeatures = self.vgg(self.Sfake_B)
            Tgram = [self.gram(fmap) for fmap in Tfeatures]
            Sgram = [self.gram(fmap) for fmap in Sfeatures]
            self.loss_G_style = 0
            for i in range(len(Tgram)):
                self.loss_G_style += self.lambda_style * F.l1_loss(Sgram[i], Tgram[i])
            Srecon, Trecon = Sfeatures[1], Tfeatures[1]
            self.loss_G_feature = self.lambda_feature * F.l1_loss(Srecon, Trecon)
            diff_i = torch.sum(torch.abs(self.Sfake_B[:, :, :, 1:] - self.Sfake_B[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(self.Sfake_B[:, :, 1:, :] - self.Sfake_B[:, :, :-1, :]))
            self.loss_G_tv = self.lambda_tv * (diff_i + diff_j)
            self.loss_G_student += self.loss_G_SSIM + self.loss_G_style + self.loss_G_feature + self.loss_G_tv
        if self.opt.lambda_CD:
            self.loss_G_CD = self.calc_CD_loss() * self.opt.lambda_CD
            self.loss_G_student += self.loss_G_CD
        self.loss_G_student.backward()

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def optimize_parameters(self, steps):
        self.optimizer_D_teacher.zero_grad()
        self.optimizer_G_teacher_w.zero_grad()
        self.optimizer_G_teacher_d.zero_grad()
        self.optimizer_G_student.zero_grad()
        self.forward()
        if steps % self.opt.n_dis == 0:
            util.set_requires_grad(self.netD_teacher, True)
            self.backward_D_teacher()
            util.set_requires_grad(self.netD_teacher, False)
            self.backward_G_teacher()
            self.optimizer_D_teacher.step()
            self.optimizer_G_teacher_w.step()
            self.optimizer_G_teacher_d.step()
        self.backward_G_student()
        self.optimizer_G_student.step()

    def load_networks(self, verbose=True):
        super(MultiTeacherDistiller, self).load_networks()

    def evaluate_model(self, step):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        self.netG_teacher_w.eval()
        self.netG_teacher_d.eval()
        S_fakes, T_fakes, names = [], [[] for _ in range(self.opt.num_teacher)],  []
        cnt = 0
        id_model_dict = {0: 'w', 1: 'd'}
        for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
            self.set_input(data_i)
            self.test()
            S_fakes.append(self.Sfake_B.cpu())
            for k in range(len(self.Tfake_Bs)):
                T_fakes[k].append(self.Tfake_Bs[k].cpu())
                for j in range(len(self.image_paths)):
                    short_path = ntpath.basename(self.image_paths[j])
                    name = os.path.splitext(short_path)[0]
                    if k == 0:
                        names.append(name)
                    if cnt < 10 * len(self.Tfake_Bs):
                        Tfake_im = util.tensor2im(self.Tfake_Bs[k][j])
                        if k == 0:
                            input_im = util.tensor2im(self.real_A[j])
                            Sfake_im = util.tensor2im(self.Sfake_B[j])
                            util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png') % name, create_dir=True)
                            util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake', '%s.png' % name), create_dir=True)
                        util.save_image(Tfake_im, os.path.join(save_dir, f'Tfake_{id_model_dict[k]}', '%s.png' %name), create_dir=True)
                        if self.opt.dataset_mode == 'aligned' and k == 0:
                            real_im = util.tensor2im(self.real_B[j])
                            util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                    cnt += 1
        fid_teachers = [get_fid(T_fakes[m], self.inception_model, self.npz, device=self.device,
                      batch_size=self.opt.eval_batch_size, tqdm_position=2) for m in range(self.opt.num_teacher)]
        fid_student = get_fid(S_fakes, self.inception_model, self.npz, device=self.device,
                      batch_size=self.opt.eval_batch_size, tqdm_position=2)
        if fid_student < self.best_fid_student:
            self.is_best = True
            self.best_fid_student = fid_student

        ret = {}
        for i in range(self.opt.num_teacher):
            ret[f'metric/fid_teacher_{id_model_dict[i]}'] = fid_teachers[i]
            if fid_teachers[i] < self.best_fid_teachers[i]:
                self.best_fid_teachers[i] = fid_teachers[i]
            ret[f'metric/fid-best_teacher_{id_model_dict[i]}'] = self.best_fid_teachers[i]
        ret['metric/fid_student'] = fid_student
        ret['metric/fid-best_student'] = self.best_fid_student
        if 'cityscapes' in self.opt.dataroot and self.opt.direction == 'BtoA':
            mIoU_teachers = [get_cityscapes_mIoU(T_fakes[m], names, self.drn_model, self.device,
                                       table_path=self.opt.table_path,
                                       data_dir=self.opt.cityscapes_path,
                                       batch_size=self.opt.eval_batch_size,
                                       num_workers=self.opt.num_threads, tqdm_position=2) for m in range(self.opt.num_teacher)]
            mIoU_student = get_cityscapes_mIoU(S_fakes, names, self.drn_model, self.device,
                                       table_path=self.opt.table_path,
                                       data_dir=self.opt.cityscapes_path,
                                       batch_size=self.opt.eval_batch_size,
                                       num_workers=self.opt.num_threads, tqdm_position=2)
            if mIoU_student > self.best_mIoU_student:
                self.is_best = True
                self.best_mIoU_student = mIoU_student
            for i in range(self.opt.num_teacher):
                ret[f'metric/mIoU_teacher_{id_model_dict[i]}'] = mIoU_teachers[i]
                if mIoU_teachers[i] > self.best_mIoU_teachers[i]:
                    self.best_mIoU_teachers[i] = mIoU_teachers[i]
                ret[f'metric/mIoU-best_teacher_{id_model_dict[i]}'] = self.best_mIoU_teachers[i]
            ret['metric/mIoU_student'] = mIoU_student
            ret['metric/mIoU-best_student'] = self.best_mIoU_student
        self.netG_teacher_w.train()
        self.netG_teacher_d.train()
        self.netG_student.train()
        return ret
