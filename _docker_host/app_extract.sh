#!/usr/bin/env bash
source shell_var.sh
rm -f ./"$img_name"_app.tar
id=$(sudo docker create $img_name)
sudo docker cp $id:/app - > ./"$img_name"_app.tar
sudo docker rm -v $id