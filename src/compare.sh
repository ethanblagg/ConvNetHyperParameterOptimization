#!/bin/bash

declare -a LEARNING_RATES           # -r
declare -a OPTIMIZERS               # -pcm
declare -a FILTER_SIZES             # -f
declare -a EPOCHS                   # -e

LEARNING_RATES=(5e-3)
OPTIMIZERS=(m)
FILTER_SIZES=(48)
EPOCHS=(3)
RUNS=2
BATCH_SIZE=64


DT_STR="$(date +"%Y-%m-%d--%H-%M-%S")"
OUT_DIR="../out/"
LOG_DIR="../log/sim/"
OUT_FILE="${DT_STR}.csv"
LOG_FILE="${DT_STR}.log"
SIM_DIR=$(pwd)
SIM_EXE="python3 main.py"

mkdir -p $OUT_DIR                        # make out dir if doesn't exist
mkdir -p $LOG_DIR


echo "learning_rate, optimizer, filter_size, epochs, loss, accuracy" > $OUT_DIR$OUT_FILE

for lr in ${LEARNING_RATES[*]}; do

    for opz in ${OPTIMIZERS[*]}; do

        for flt in ${FILTER_SIZES[*]}; do

            for ep in ${EPOCHS[*]}; do
                
                for ((i=1; i<=RUNS; i++)); do
                    
                    echo "Run $i/$RUNS for:"
                    echo "$SIM_EXE -r$lr -f$flt -e$ep -b$BATCH_SIZE -$opz"
                    CMD_OUTPUT=$($SIM_EXE -r$lr -f$flt -e$ep -b$BATCH_SIZE -$opz | tee /dev/tty)
                    echo "$CMD_OUTPUT" >> $LOG_DIR$LOG_FILE

                    # 645/645 [==============================] - 39s 60ms/step - loss: 1.8330 - accuracy: 0.5587
                    res_line=$(echo "$CMD_OUTPUT" | egrep -o "323/323.*$") # grep til \n bc no newlines for 1-644, need to drop
                    loss=$(echo "$res_line" | egrep -o "loss: [[:digit:]]+.[[:digit:]]+" | egrep -o "[[:digit:]]+.[[:digit:]]+")
                    acc=$(echo "$res_line" | egrep -o "accuracy: [[:digit:]]+.[[:digit:]]+" | egrep -o "[[:digit:]]+.[[:digit:]]+")
                   
                    echo "$lr,$opz,$flt,$ep,$loss,$acc" >> $OUT_DIR$OUT_FILE

                done

            done

        done

    done

done



