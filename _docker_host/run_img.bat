call shell_var.bat
docker run -p 0.0.0.0:80:80 -it ^
    --shm-size="256m" ^
    -w /app --mount type=bind,src="%cd%"\..\data,target=/app/data ^
    %img_name% ^
    sh %exe_file%