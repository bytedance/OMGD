#!/usr/bin/env bash
python test.py --dataroot database/edges2shoes-r \
  --results_dir results/resnet_pix2pix/edges2shoes-r/S16 \
  --ngf 16 --netG mobile_resnet_9blocks \
  --restore_G_path checkpoints/resnet_pix2pix/edges2shoes/best_net_G16.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --need_profile --num_test 3000
