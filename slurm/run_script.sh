#! /bin/bash
if [ -z "$RRC_ROOT" ]
then
      echo "Please set \$RRC_ROOT (e.g., export RRC_ROOT=/path/to/your/rrc_phase_2)"
      exit 1
fi

if [ -z "$RRC_IMAGE" ]
then
      echo "Please set \$RRC_IMAGE to the path of your singularity image"
      exit 1
fi
cd $RRC_ROOT
echo `date`
mounts=`python slurm/build_ws.py $@`
echo ${mounts}
echo `date`
mkdir -p /scratch/cbschaff
rand_char=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5`
img_file="/scratch/cbschaff/${rand_char}.sif"
echo ${img_file}
cp $RRC_IMAGE ${img_file}
if [ $? -eq 0 ]
then
    echo `date`
    echo "Using scratch"
    singularity exec --contain --nv -B ${mounts} ${img_file} bash -c \
        ". /setup.bash; . /ws/devel/setup.bash; timeout -s SIGINT -k 0.1h --foreground 3.5h python3 /ws/src/usercode/python/code/train_ppo.py /logdir/singularity_config.yaml"
else
    echo `date`
    echo "Not using scratch"
    singularity exec --contain --nv -B ${mounts} $RRC_IMAGE bash -c \
        ". /setup.bash; . /ws/devel/setup.bash; timeout -s SIGINT -k 0.1h --foreground 3.5h python3 /ws/src/usercode/python/code/train_ppo.py /logdir/singularity_config.yaml"
fi
echo `date`
rm ${img_file}
echo `date`
