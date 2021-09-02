#!/usr/bin/env bash
python distill.py --dataroot database/edges2shoes-r \
  --gpu_ids 0 --print_freq 100 --n_share 5 \
  --lambda_CD 1e1 \
  --distiller multiteacher \
  --log_dir logs/unet_pix2pix/edges2shoes-r/distill_S8 \
  --batch_size 4 --num_teacher 2 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --teacher_ngf_w 48 --teacher_ngf_d 12 --student_ngf 8  --norm batch --ndf 96 \
  --teacher_netG_w unet_256 --teacher_netG_d unet_deepest_256 --netD multi_n_layers \
  --nepochs 100 --nepochs_decay 100 --n_dis 1 \
  --AGD_weights 1e1,1e4,1e1,1e-5