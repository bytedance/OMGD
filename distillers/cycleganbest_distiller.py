import ntpath
import os

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.parallel import gather, parallel_apply, replicate
from tqdm import tqdm

from metric import get_fid, get_cityscapes_mIoU
from utils import util
from utils.vgg_feature import VGGFeature
from .base_cycleganbest_distiller import BaseCycleganBestDistiller
from models.modules import pytorch_ssim
from utils.logger import Logger
import time

class CycleganBestDistiller(BaseCycleganBestDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(CycleganBestDistiller, CycleganBestDistiller).modify_commandline_options(parser, is_train)
        parser.add_argument('--AGD_weights', type=str, default='1e1, 1e4, 1e1, 1e-5', help='weights for losses in AGD mode')
        parser.add_argument('--n_dis', type=int, default=1, help='iter time for student before update teacher')
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(CycleganBestDistiller, self).__init__(opt)
        self.best_fid_teacher_A, self.best_fid_teacher_B, self.best_fid_student = 1e9, 1e9, 1e9
        self.best_mIoU_teacher, self.best_mIoU_student = -1e9, -1e9
        self.fids_teacher_A, self.fids_teacher_B, self.fids_student, self.mIoUs_teacher, self.mIoUs_student = [], [], [], [], []
        # weights for AGD mood
        loss_weight = [float(char) for char in opt.AGD_weights.split(',')]
        self.lambda_SSIM = loss_weight[0]
        self.lambda_style = loss_weight[1]
        self.lambda_feature = loss_weight[2]
        self.lambda_tv = loss_weight[3]
        self.vgg = VGGFeature().to(self.device)
        self.logger = Logger(self.opt)

    def forward(self):
        self.fake_B = self.netG_teacher_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_teacher_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_teacher_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_teacher_A(self.fake_A)  # G_A(G_B(B))

    def backward_G_Teacher(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_teacher_A(self.real_B)
            self.loss_G_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_teacher_B(self.real_A)
            self.loss_G_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_G_idt_A = 0
            self.loss_G_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_teacher_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_teacher_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_G_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_G_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_cycle_A + self.loss_G_cycle_B + self.loss_G_idt_A + self.loss_G_idt_B
        self.loss_G.backward()

    def calc_CD_loss(self):
        losses = []
        mapping_layers = self.mapping_layers[self.opt.teacher_netG]
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

    def backward_G_Student(self):
        self.fake = self.netG_teacher_A(self.real_A)
        self.Sfake = self.netG_student(self.real_A)
        teacher_image = self.fake.detach()
        ssim_loss = pytorch_ssim.SSIM()
        self.loss_G_SSIM = self.lambda_SSIM * (1 - ssim_loss(self.Sfake, teacher_image))
        Tfeatures = self.vgg(teacher_image)
        Sfeatures = self.vgg(self.Sfake)
        Tgram = [self.gram(fmap) for fmap in Tfeatures]
        Sgram = [self.gram(fmap) for fmap in Sfeatures]
        self.loss_G_style = 0
        for i in range(len(Tgram)):
            self.loss_G_style += self.lambda_style * F.l1_loss(Sgram[i], Tgram[i])
        Srecon, Trecon = Sfeatures[1], Tfeatures[1]
        self.loss_G_feature = self.lambda_feature * F.l1_loss(Srecon, Trecon)
        diff_i = torch.sum(torch.abs(self.Sfake[:, :, :, 1:] - self.Sfake[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(self.Sfake[:, :, 1:, :] - self.Sfake[:, :, :-1, :]))
        self.loss_G_tv = self.lambda_tv * (diff_i + diff_j)
        self.loss_G_student = self.loss_G_SSIM + self.loss_G_style + self.loss_G_feature + self.loss_G_tv
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
        self.forward()
        self.optimizer_D_teacher.zero_grad()
        self.optimizer_G_teacher.zero_grad()
        self.set_requires_grad([self.netD_teacher_A, self.netD_teacher_B], False)  # Ds require no gradients when optimizing Gs
        self.backward_G_Teacher()  # calculate gradients for G_A and G_B
        self.optimizer_G_teacher.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_teacher_A, self.netD_teacher_B], True)
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D_teacher.step()  # update D_A and D_B's weights

    def optimize_student_parameters(self):
        for epoch in range(self.opt.n_dis * self.opt.save_epoch_freq):
            self.student_epoch += 1
            for i, data_i in enumerate(tqdm(self.student_dataloader, desc='Batch      ', position=1, leave=False)):
                iter_start_time = time.time()
                self.student_steps += 1
                self.set_single_input(data_i)
                self.optimizer_G_student.zero_grad()
                self.backward_G_Student()
                self.optimizer_G_student.step()
                if self.student_steps % self.opt.print_freq == 0:
                    losses = self.get_current_losses()
                    self.logger.print_current_errors(self.student_epoch, self.student_steps, losses, time.time() - iter_start_time)
                if self.student_steps % self.opt.save_latest_freq == 0:
                    start_time = time.time()
                    metrics = self.evaluate_student_model(self.student_steps)
                    self.logger.print_current_metrics(self.student_epoch, self.student_steps, metrics, time.time() - start_time)
                    self.save_student_networks('lastet')
                    self.logger.print_info('Saving the latest student model (epoch %d, total_steps %d)' % (self.student_epoch, self.student_steps))
                    if self.is_best:
                        self.save_student_networks('iter%d' % self.student_steps)
            if self.student_epoch % self.opt.save_epoch_freq == 0 or epoch == self.opt.n_dis * self.opt.save_epoch_freq - 1:
                start_time = time.time()
                metrics = self.evaluate_student_model(self.student_steps)
                self.logger.print_current_metrics(self.student_epoch, self.student_steps, metrics, time.time() - start_time)
                self.save_student_networks('lastet')
                self.logger.print_info(
                    'Saving the student model at the end of epoch %d, iters %d' % (self.student_epoch, self.student_steps))
                self.save_student_networks(self.student_epoch)

    def test_single_side(self, direction):
        generator = getattr(self, 'netG_teacher_%s' % direction[0])
        with torch.no_grad():
            self.fake_B = generator(self.real_A)

    def evaluate_model(self, step):

        save_filename = 'best_net_G_teacher_A.pth'
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.ret = {}
        self.netG_teacher_A.eval()
        self.netG_teacher_B.eval()
        for direction in ['AtoB', 'BtoA']:
            eval_dataloader = getattr(self, 'eval_dataloader_' + direction)
            T_fakes, names = [], []
            cnt = 0
            for i, data_i in enumerate(tqdm(eval_dataloader, desc='Eval    ', position=2, leave=False)):
                self.set_single_input(data_i)
                self.test_single_side(direction)
                T_fakes.append(self.fake_B.cpu())
                for j in range(len(self.image_paths)):
                    short_path = ntpath.basename(self.image_paths[j])
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
                    if cnt < 10:
                        input_im = util.tensor2im(self.real_A[j])
                        Tfake_im = util.tensor2im(self.fake_B[j])
                        util.save_image(input_im, os.path.join(save_dir, direction, 'input', '%s.png') % name,
                                        create_dir=True)
                        util.save_image(Tfake_im, os.path.join(save_dir, direction, 'fake', '%s.png' % name),
                                        create_dir=True)
                    cnt += 1
            suffix = direction[-1]
            fid_teacher = get_fid(T_fakes, self.inception_model, getattr(self, 'npz_%s' % direction[-1]),
                                  device=self.device, batch_size=self.opt.eval_batch_size, tqdm_position=2)
            if getattr(self, 'best_fid_teacher_%s' % suffix) > fid_teacher:
                if getattr(self, 'best_fid_teacher_%s' % suffix) > fid_teacher:
                    setattr(self, 'best_fid_teacher_%s' % suffix, fid_teacher)
                save_path = os.path.join(self.save_dir, save_filename)
                self.save_net(self.netG_teacher_A, save_path)
            self.ret['metric/fid_teacher_%s' % suffix] = fid_teacher
            self.ret['metric/fid-best_teacher_%s' % suffix] = getattr(self, 'best_fid_teacher_%s' % suffix)
        self.netG_teacher_A.train()
        self.netG_teacher_B.train()
        return self.ret

    def evaluate_student_model(self, step):
        self.is_best = False
        self.netG_student.eval()
        save_dir = os.path.join(self.opt.log_dir, 'eval_student', str(step))
        os.makedirs(save_dir, exist_ok=True)
        direction = self.opt.direction
        eval_dataloader = getattr(self, 'eval_dataloader_' + direction)
        S_fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(eval_dataloader, desc='Eval    ', position=2, leave=False)):
            self.set_single_input(data_i)
            with torch.no_grad():
                self.Sfake_B = self.netG_student(self.real_A)
            S_fakes.append(self.Sfake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10:
                    input_im = util.tensor2im(self.real_A[j])
                    Sfake_im = util.tensor2im(self.Sfake_B[j])
                    util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake', '%s.png' % name),
                                    create_dir=True)
                    util.save_image(input_im, os.path.join(save_dir, direction, 'input', '%s.png') % name,
                                    create_dir=True)
                cnt += 1

        fid_student = get_fid(S_fakes, self.inception_model, getattr(self, 'npz_%s' % direction[-1]),
                              device=self.device, batch_size=self.opt.eval_batch_size, tqdm_position=2)
        if fid_student < self.best_fid_student:
            self.is_best = True
            self.best_fid_student = fid_student
        self.ret['metric/fid_student'] = fid_student
        self.ret['metric/fid-best_student'] = self.best_fid_student
        self.netG_student.train()
        return self.ret

    def load_best_teacher(self):
        save_filename = 'best_net_G_teacher_A.pth' if self.opt.direction == 'AtoB' else 'best_net_G_teacher_B.pth'
        util.load_network(self.netG_teacher_A, os.path.join(self.save_dir, save_filename))
        self.netG_teacher_A.train()

    def load_latest_teacher(self):
        latest_filename = 'latest_net_G_teacher_A.pth' if self.opt.direction == 'AtoB' else 'latest_net_G_teacher_B.pth'
        util.load_network(self.netG_teacher_A, os.path.join(self.save_dir, 'teacher', latest_filename))
        self.netG_teacher_A.train()


