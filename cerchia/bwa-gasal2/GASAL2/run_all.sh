#!/bin/bash
./configure.sh /home/usuaris/lmaiese/cuda-11.0
make GPU_SM_ARCH=sm_70 MAX_SEQ_LEN=300 N_CODE=4 N_PENALTY=1
cd test_prog
make $1 
sha256sum -c crc_out.log.crc
cd ..
