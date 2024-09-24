call shell_var.bat
FOR /F "tokens=* USEBACKQ" %%F IN (`docker ps --filter ancestor^=%img_name% --format "{{.ID}}"`) DO (
  SET id=%%F
)
docker stop %id%
FOR /F "tokens=* USEBACKQ" %%F IN (`docker images %img_name% --format "{{.ID}}"`) DO (
  SET id=%%F
)
docker rmi -f %id%
docker system prune -f
docker load --input %img_name%.tar
docker system prune -f
