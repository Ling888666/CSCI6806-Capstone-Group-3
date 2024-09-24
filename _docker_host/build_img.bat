call shell_var.bat
call stop_img.bat
IF EXIST .\%img_name%.tar DEL /F .\%img_name%.tar
docker build --platform linux/amd64 -t %img_name% ..
docker system prune -f
