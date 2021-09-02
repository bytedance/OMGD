#!/usr/bin/env bash
python distill.py --dataroot database/cityscapes \
  --gpu_ids 4 --print_freq 100 \
  --distiller multiteacher \
  --log_dir logs/resnet_pix2pix/cityscapes/distill \
  --num_teacher 2 --n_share 4 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --teacher_ngf_w 64 --teacher_ngf_d 16 --student_ngf 16 \
  --teacher_netG_w mobile_resnet_9blocks --teacher_netG_d mobile_deepest_resnet \
  --student_netG mobile_resnet_9blocks --netD multi_n_layers \
  --nepochs 300 --nepochs_decay 450 --n_dis 3 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path  database/cityscapes-origin \
  --table_path  datasets/table.txt \
  --direction BtoA  --AGD_weights 1e1,1e4,1e1,1e-5