call shell_var.bat
call build_img.bat
IF EXIST .\%img_name%_app.tar DEL /F .\%img_name%_app.tar
call extract_img.bat