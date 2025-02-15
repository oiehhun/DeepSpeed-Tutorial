#!/bin/sh

echo 'Deepspeed Tutorial Satart'

DEEPSPEED_CONFIG="/root/deepspeed_tutorial/deepspeed_config_zero3.json"

deepspeed --master_port 10000 \
          --include localhost:0,1,2,3,4,5,6,7 \
    main.py \
        --deepspeed_config $DEEPSPEED_CONFIG