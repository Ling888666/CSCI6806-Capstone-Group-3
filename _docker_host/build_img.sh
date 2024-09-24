#!/usr/bin/env bash
source shell_var.sh
source stop_img.sh
rm -f ./$img_name.tar
sudo docker build --platform linux/amd64 -t $img_name ..
sudo docker system prune -f
