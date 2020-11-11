#!/bin/bash
if (( $# != 4 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <log directory> <singularity image> <job id> <job status>"
    exit 1
fi
logdir=$1
image=$2
jobid=$3
status=$4

if [ -z "$RRC_ROOT" ]
then
      echo "Please set \$RRC_ROOT (e.g., export RRC_ROOT=/path/to/your/rrc_phase_2)"
      exit 1
fi

dir="$RRC_ROOT/log_manager"

if [ -d ${logdir}/${jobid} ]; then
    exit
fi

if [ ! ${status} == 'C' ]; then
    exit
fi

rm -r ${logdir}/${jobid}

bash ${dir}/download_logs.sh ${jobid} ${logdir}
singularity run --nv -B /share ${image} python3 ${dir}/replay_scripts/compute_reward.py ${logdir}/${jobid}
bash ${dir}/make_plots.sh ${image} ${logdir}/${jobid}
bash ${dir}/make_video.sh ${image} ${logdir}/${jobid}
