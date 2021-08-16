#!/usr/bin/env bash
python test.py --dataroot  database/edges2shoes-r \
  --results_dir  results/unet_pix2pix/edges2shoes-r/S12 \
  --ngf 12 --netG unet_256 --norm batch \
  --restore_G_path checkpoints/unet_pix2pix/edges2shoes/best_net_G12.pth \
  --real_stat_path  real_stat/edges2shoes-r_B.npz \
  --need_profile --num_test 3000 --phase val
