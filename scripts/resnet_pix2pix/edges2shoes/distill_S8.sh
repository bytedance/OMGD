#!/usr/bin/env bash
python distill.py --dataroot database/edges2shoes-r \
  --gpu_ids 1 \
  --lambda_CD 1e3 \
  --batch_size 4 \
  --distiller multiteacher \
  --log_dir logs/resnet_pix2pix/edges2shoes-r/distill_S8 \
  --num_teacher 2 --n_share 3 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --teacher_ngf_w 48 --teacher_ngf_d 12 --student_ngf 8 --ndf 96 \
  --teacher_netG_w mobile_resnet_9blocks --teacher_netG_d mobile_deepest_resnet \
  --student_netG mobile_resnet_9blocks --netD multi_n_layers \
  --nepochs 100 --nepochs_decay 100 --n_dis 1 \
  --AGD_weights 1e1,1e4,1e1,1e-5