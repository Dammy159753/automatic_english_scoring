#!/bin/bash
#source /etc/profile
#gunicorn -b 0.0.0.0:10000 AES_server:app
CUDA_VISIBLE_DEVICES=$1 nohup gunicorn -w 1 -t 600 -b 0.0.0.0:$2 AES_server:app > run.log 2>&1 &
tail -f /dev/null