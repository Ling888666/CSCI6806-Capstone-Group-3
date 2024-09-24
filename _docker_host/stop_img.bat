call shell_var.bat
FOR /F "tokens=* USEBACKQ" %%F IN (`docker ps --filter ancestor^=%img_name% --format "{{.ID}}"`) DO (
  SET id=%%F
)
docker stop %id%