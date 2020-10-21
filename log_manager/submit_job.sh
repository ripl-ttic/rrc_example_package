#!/bin/bash
# prompt for username and password (to avoid having user credentials in the
# bash history)
dir=`dirname $0`
username=`cat ${dir}/user.txt | head -n 1`
password=`cat ${dir}/user.txt | sed -n '2 p'`
# there is no automatic new line after the password prompt
echo


# URL to the webserver at which the recorded data can be downloaded
hostname=robots.real-robot-challenge.com
data_url=https://${hostname}/output/${username}/data


# Check if the file/path given as argument exists in the "data" directory of
# the user
function curl_check_if_exists()
{
    local filename=$1

    http_code=$(curl -sI --user ${username}:${password} -o /dev/null -w "%{http_code}" ${data_url}/${filename})

    case "$http_code" in
        200|301) return 0;;
        *) return 1;;
    esac
}

function submit_job()
{
    echo "Submit job"
    submit_result=$(ssh -T -i sshkey ${username}@${hostname} <<<submit)
    job_id=$(echo "${submit_result}" | cut -d' ' -f 6 | grep -oE '[0-9]+')
    if [ $? -ne 0 ]
    then
        echo "Failed to submit job.  Output:"
        echo "${submit_result}"
        echo "Job ID ${job_id}"
        exit 1
    fi
    echo "Submitted job with ID ${job_id}"

}

function wait_for_job_start()
{
    echo "Wait for job to be started"
    local job_started=0
    # wait for the job to start (abort if it did not start after half an hour)
    for (( i=0; i<30; i++))
    do
        # Do not poll more often than once per minute!
        sleep 60

        # wait for the job-specific output directory
        if curl_check_if_exists $1
        then
            local job_started=1
            break
        fi
        date
    done
    if (( ${job_started} == 0 ))
    then
        echo "Job did not start."
        exit 1
    fi

    echo "Job is running.  Wait until it is finished"
}

function wait_for_job_finish()
{
    # if the job did not finish 10 minutes after it started, something is
    # wrong, abort in this case
    local job_finished=0
    for (( i=0; i<15; i++))
    do
        # Do not poll more often than once per minute!
        sleep 60

        # report.json is explicitly generated last of all files, so once this
        # exists, the job has finished
        if curl_check_if_exists $1/report.json
        then
            local job_finished=1
            echo "The job has finished."
            break
        fi
        date
    done
}

submit_job
wait_for_job_start ${job_id}
wait_for_job_finish ${job_id}
