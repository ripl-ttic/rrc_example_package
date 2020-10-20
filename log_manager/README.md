# Log Manager

This code downloads the logs of all jobs executed on the real TriFinger robots and generates videos from the camera logs.

In order to use this code, you must add two files in this directory:

1) user.txt: This should contain only the username and password (each on their own line) used to download the data 

2) sshkey: This should contain the ssh private key used to log into the submission system


Calling run.sh will download all new log data and create videos.
