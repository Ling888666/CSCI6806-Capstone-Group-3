#! /bin/bash
source shell_var.sh
sudo docker run -p 0.0.0.0:80:80 \
    --shm-size="256m" \
    -w /app --mount type=bind,src=${PWD}/../data,target=/app/data \
    $img_name \
    sh $exe_file
