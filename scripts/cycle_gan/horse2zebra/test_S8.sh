#!/usr/bin/env bash
python test.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir  results/cycle_gan/horse2zebra/S8 \
  --ngf 8 --netG mobile_resnet_9blocks \
  --restore_G_path checkpoints/cyclegan/horse2zebra/best_net_G8.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz