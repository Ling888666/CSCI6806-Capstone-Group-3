call shell_var.bat
IF EXIST .\%img_name%_app.tar DEL /F .\%img_name%_app.tar
IF EXIST .\%img_name%.tar DEL /F .\%img_name%.tar