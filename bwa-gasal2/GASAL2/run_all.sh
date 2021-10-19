#!/bin/bash
./configure.sh /usr/local/cuda10.1
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=300 N_CODE=4 N_PENALTY=1
cd test_prog
make $1 
sha256sum -c crc_out.log.crc
cd ..
