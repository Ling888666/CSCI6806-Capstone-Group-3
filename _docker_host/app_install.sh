#!/usr/bin/env bash
source shell_var.sh
tar xf ./"$img_name"_app.tar -C .
id=$(sudo docker run -d -t $img_name rm -rf /app)
sudo docker cp ./app $id:/
sudo docker commit $id $img_name
sudo docker rm -v $id
rm -rf ./app
sudo docker system prune -f