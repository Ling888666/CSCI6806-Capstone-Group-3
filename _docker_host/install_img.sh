#!/usr/bin/env bash
source shell_var.sh
sudo docker stop $(sudo docker ps --filter ancestor=$img_name --format "{{.ID}}")
sudo docker rmi -f $(sudo docker images $img_name --format "{{.ID}}")
sudo docker system prune -f
sudo docker load --input $img_name.tar
sudo docker system prune -f