#!/bin/bash
#source /etc/profile
#nohup gunicorn --timeout 600 -b 0.0.0.0:10001 AES_server:app > log.txt 2>&1 &
#tail -f /dev/null
CUDA_VISIBLE_DEVICES=$1 gunicorn -w 1 -t 600 -b 0.0.0.0:$2 --chdir .. AES_server:app