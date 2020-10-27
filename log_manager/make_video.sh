#!/bin/bash
if (( $# != 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <singularity image> <log directory>"
    exit 1
fi
image=$1
logdir=$2
dir=`dirname $0`
singularity run --nv ${image} python3 ${dir}/replay_scripts/replay.py ${logdir} ${logdir}/video.avi
ffmpeg -i ${logdir}/video.avi ${logdir}/video.webm
rm ${logdir}/video.avi
