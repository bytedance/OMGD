#!/usr/bin/env bash
python latency.py --dataroot /opt/tiger/renyuxi_run_on_hdfs/database/edges2shoes-r \
  --gpu_ids 0 \
  --results_dir results/resnet_pix2pix/cityscapes/teacher \
  --ngf 64 --netG resnet_9blocks \
  --restore_G_path pretrained/pix2pix/edges2shoes-r/full/latest_net_G.pth \
  --real_stat_path /opt/tiger/renyuxi_run_on_hdfs/real_stat/edges2shoes-r_B.npz \
  --need_profile