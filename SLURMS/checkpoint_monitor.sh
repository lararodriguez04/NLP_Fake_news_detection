#!/bin/bash

CHECKPOINT_DIR="/home/lnlpG08/nlp/RESULTS/grpo_explanation/grpo_model"

echo "$(date) - Checkpoint monitor started"

while true; do
    CHECKPOINTS=$(ls -td $CHECKPOINT_DIR/checkpoint-*/ 2>/dev/null | while read dir; do
        if [ -f "${dir}trainer_state.json" ]; then
            echo $dir
        fi
    done)

    COUNT=$(echo "$CHECKPOINTS" | grep -c "checkpoint" 2>/dev/null || echo 0)

    if [ "$COUNT" -gt 1 ]; then
        OLDEST=$(echo "$CHECKPOINTS" | tail -n +2)
        while IFS= read -r dir; do
            if [ -n "$dir" ]; then
                echo "$(date) - Removing old checkpoint: $dir"
                rm -rf "$dir"
            fi
        done <<< "$OLDEST"
    fi

    rm -rf ~/.cache/huggingface/metrics/

    sleep 60
done
