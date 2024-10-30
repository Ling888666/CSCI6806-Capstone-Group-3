### What's this?

Capstone Project! With Docker! Identical developing environments for every team member! Yay~~~

### Requirements:
1. Install [Docker](https://docs.docker.com/engine/install/) on your system.
2. Pull everything here into a local directory.
3. Put project __*.csv__ data files under the __./data__ directory.
4. Play with command scripts under **./_docker_host** directory.

    They should work fine under *Windows*(Post Windows 10 Build 17063),*Ubuntu* and *MacOS*.

### Command Usage:
1. Commands need to be executed inside your system corresponding terminal from within the **./_docker_host** directory. In case of *Windows*, run the __*.bat__ variants, otherwise, go for the __*.sh__ ones.

    You may need to give running permissions to the files under **./_docker_host** directory before hand.
    ```
    find ./_docker_host -name "*.sh" -execdir chmod u+x {} +
    ```

2. #### *build_img*

    Execute this to build a runnable docker image of the project inside your system, you need this for other commands to work.

3. #### *run_img*

    Run the project program from the built image.

4. #### *debug_run*

    Run changed python codes directly without rebuilding the docker image. This only works inside a developing environments.

5. #### *extract_img*
  
    Extract the image __*.tar__ file for deployment. After running this, you can copy **./_docker_host** and __./data__ directories to another docker ready computer and run *install_img* from there.

6. #### *install_img*
  
    Install and override any local image from the extracted __*.tar__ file under **./_docker_host** directory. After installation, *run_img* should be functional.

7. #### *release_all*
  
    A convenient script chaining *build_img* and *extract_img* together.

8. #### *clear_all*
  
    Remove any generated __*.tar__ files.