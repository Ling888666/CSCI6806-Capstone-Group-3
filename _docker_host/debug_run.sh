#!/usr/bin/env bash
source shell_var.sh
sudo docker run --platform linux/amd64 -p 0.0.0.0:80:80 -it \
   -w /app --mount type=bind,src=${PWD}/../,target=/app \
   $img_name \
   sh $exe_file