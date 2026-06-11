#!/bin/bash

SCRIPT="/home/lnlpG08/nlp/RL_SUMMARIZATION.py"
LOG="/home/lnlpG08/nlp/output_rl.log"
CHECKPOINT_DIR="/home/lnlpG08/nlp/RESULTS/grpo_explanation/grpo_model"

while true; do
    echo "Launching RL training..." >> $LOG

    python3 $SCRIPT >> $LOG 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully." >> $LOG
        break
    fi

    echo "Process failed with code $EXIT_CODE. Checking for disk issues..." >> $LOG

    if grep -q "Disk quota exceeded\|unexpected pos\|enforce fail" $LOG; then
        echo "Disk error detected. Cleaning old checkpoints..." >> $LOG

        LAST=$(ls -td $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | head -1)

        if [ -n "$LAST" ]; then
            for dir in $CHECKPOINT_DIR/checkpoint-*/; do
                if [ "$dir" != "$LAST/" ]; then
                    echo "Removing $dir" >> $LOG
                    rm -rf "$dir"
                fi
            done

            STEP=$(basename $LAST | sed 's/checkpoint-//')
            echo "Resuming from $LAST (step $STEP)" >> $LOG
            sed -i "s|checkpoint-[0-9]*\"|checkpoint-$STEP\"|g" $SCRIPT
        fi

        sleep 5
    else
        echo "Unknown error, stopping." >> $LOG
        break
    fi
done
