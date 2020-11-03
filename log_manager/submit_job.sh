#!/bin/bash

. ./funcs.sh
wait_for_job_submission
submit_job
wait_for_job_start ${job_id}
wait_for_job_finish ${job_id}
