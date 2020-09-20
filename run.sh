# !/bin/bash

appr="dqn"
env_path=$PWD"/env/"
_now=$(date +"%Y-%m-%d-%H%M%S")
_file="./${appr}_training_${_now}.out"

args="--appr="$appr" --envpth="$env_path

nohup python main.py $args > $_file 2>&1 &
