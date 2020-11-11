#!/bin/bash
image=$1
logdir=$2
dir=`dirname $0`
singularity run -B /share ${image} rosrun robot_fingers robot_log_dat2csv.py ${logdir}/robot_data.dat ${logdir}/robot_data.csv
singularity run -B /share ${image} python3 ${dir}/plot_scripts/finger_position.py ${logdir}/robot_data.csv
singularity run -B /share ${image} python3 ${dir}/plot_scripts/plot.py ${logdir}/robot_data.csv ${logdir}/plot_action_repeat.pdf status_action_repetitions
