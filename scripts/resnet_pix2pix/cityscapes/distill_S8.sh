#!/usr/bin/env bash
python distill.py --dataroot database/cityscapes \
  --gpu_ids 5 --print_freq 100 \
  --distiller multiteacher \
  --log_dir logs/resnet_pix2pix/cityscapes/distill_S8 \
  --num_teacher 2 --n_share 3 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --teacher_ngf_w 48 --teacher_ngf_d 12 --student_ngf 8 --ndf 96 \
  --teacher_netG_w mobile_resnet_9blocks --teacher_netG_d mobile_deepest_resnet \
  --student_netG mobile_resnet_9blocks --netD multi_n_layers \
  --nepochs 300 --nepochs_decay 450 --n_dis 3 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path  database/cityscapes-origin \
  --table_path  datasets/table.txt \
  --direction BtoA  --AGD_weights 1e1,1e4,1e1,1e-5