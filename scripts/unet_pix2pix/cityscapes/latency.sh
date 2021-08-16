#!/usr/bin/env bash
python latency.py --dataroot database/cityscapes \
  --gpu_ids -1 \
  --results_dir results/unet_pix2pix/cityscapes/S16 \
  --ngf 16 --netG unet_256 --norm batch \
  --restore_G_path checkpoints/unet_pix2pix/cityscapes/best_net_G16.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --need_profile