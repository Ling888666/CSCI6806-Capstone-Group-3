call shell_var.bat
IF EXIST .\data\%img_name%.tar DEL /F .\data\%img_name%.tar
docker save -o ./%img_name%.tar %img_name%:latest