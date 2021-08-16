#!/usr/bin/env bash
python test.py --dataroot database/summer2winter/valA \
  --dataset_mode single \
  --results_dir  results/cycle_gan/summer2winter/S12 \
  --ngf 12 --netG mobile_resnet_9blocks \
  --restore_G_path checkpoints/cyclegan/summer2winter/best_net_G12.pth \
  --need_profile \
  --real_stat_path real_stat/summer2winter_B.npz
