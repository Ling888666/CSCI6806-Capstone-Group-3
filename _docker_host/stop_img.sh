#!/usr/bin/env bash
source shell_var.sh
sudo docker stop $(sudo docker ps --filter ancestor=$img_name --format "{{.ID}}")