#!/bin/bash
if (( $# != 1 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <mount_dir>"
    exit 1
fi
mount_dir=$1
docker run -p 8000:80 -v ${mount_dir}:/opt/www/files --rm -it -d mohamnag/nginx-file-browser
