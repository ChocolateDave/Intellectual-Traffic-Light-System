# !/bin/bash

appr = "dqn"
env_path = $PWD"/env/"
cur_time = $(date -d "today" +"%Y%m%d_%H%M%S")

args = "--appr="$appr",--envpth="$env_path

nohup python main.py $args > $cur_time.out 2>&1 &