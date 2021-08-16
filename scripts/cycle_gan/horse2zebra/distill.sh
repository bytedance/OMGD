#!/usr/bin/env bash
python distill.py --dataroot database/horse2zebra \
  --gpu_ids 0 \
  --distiller cycleganbest \
  --log_dir logs/cycle_gan/horse2zebra/distill_S16 \
  --real_stat_A_path real_stat/horse2zebra_A.npz \
  --real_stat_B_path real_stat/horse2zebra_B.npz \
  --teacher_ngf 64 --student_ngf 16 \
  --lambda_CD 5e2 \
  --teacher_netG mobile_resnet_9blocks --student_netG mobile_resnet_9blocks \
  --nepochs 100 --nepochs_decay 100 --n_dis 4 \
  --save_latest_freq 25000 --save_epoch_freq 10 \
  --AGD_weights 1e1,1e4,1e1,1e-5