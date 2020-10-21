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

if [ -d ${logdir}/${jobid} ]; then
    exit
fi

if [ ! ${status} == 'C' ]; then
    exit
fi

bash download_logs.sh ${jobid} ${logdir}
bash make_video.sh ${image} ${logdir}/${jobid}
