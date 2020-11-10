#!/usr/bin/env bash

num_job_sequence=10

if (( $# == 0 )); then
    echo "usage: submit_jobs_by_commits.sh <branch 0> <branch 1> ... <branch n>"
    echo "As default, 10 sequential jobs are submitted for each branch."
fi

[[ ! -f ../roboch.json ]] && echo "roboch.json is not found at ../"

for var in "$@"
do
    echo "======== Setting 'branch' in roboch.json to be ${var} ========"
    echo 'pre-config:'
    cat ../roboch.json
    org_branch_txt=`grep "branch" ../roboch.json`
    branch_txt="  \"branch\": \"${var}\","
    sed -i '' "s/  \"branch\":.*/${branch_txt}/" ../roboch.json
    echo 'after-config:'
    cat ../roboch.json
    ./submit_jobs_in_sequence.sh ${num_job_sequence}
    sed -i '' "s/  \"branch\":.*/${org_branch_txt}/" ../roboch.json
done
