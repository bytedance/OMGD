#!/usr/bin/env bash
python distill.py --dataroot database/edges2shoes-r \
  --gpu_ids 5 \
  --distiller multiteacher \
  --lambda_CD 1e2 \
  --log_dir logs/resnet_pix2pix/edges2shoes-r/distill \
  --num_teacher 2 --n_share 3 --batch_size 4 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --teacher_ngf_w 64 --teacher_ngf_d 16 --student_ngf 16 \
  --teacher_netG_w mobile_resnet_9blocks --teacher_netG_d mobile_deepest_resnet \
  --student_netG mobile_resnet_9blocks --netD multi_n_layers \
  --nepochs 100 --nepochs_decay 100 --n_dis 1 \
  --AGD_weights 1e1,1e4,1e1,1e-5