call shell_var.bat
tar xf %img_name%_app.tar -C .
FOR /F "tokens=* USEBACKQ" %%F IN (`docker run -d -t %img_name% rm -rf /app`) DO (
  SET id=%%F
)
docker cp .\app %id%:/
docker commit %id% %img_name%
docker rm -v %id%
RMDIR .\app /S /Q
docker system prune -f

