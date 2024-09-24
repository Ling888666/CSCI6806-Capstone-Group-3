#!/usr/bin/env bash
source shell_var.sh
rm -f ./data/$img_name.tar
sudo docker save -o ./$img_name.tar $img_name:latest