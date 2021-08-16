#!/usr/bin/env bash
python latency.py --dataroot database/edges2shoes-r \
  --gpu_ids 0 \
  --results_dir results/unet_pix2pix/edges2shoes-r/S16 \
  --ngf 16 --netG unet_256 --norm batch \
  --restore_G_path checkpoints/unet_pix2pix/edges2shoes/best_net_G16.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --need_profile