call shell_var.bat
IF EXIST .\%img_name%_app.tar DEL /F .\%img_name%_app.tar
FOR /F "tokens=* USEBACKQ" %%F IN (`docker create %img_name%`) DO (
  SET id=%%F
)
docker cp %id%:/app - > .\%img_name%_app.tar
docker rm -v %id%