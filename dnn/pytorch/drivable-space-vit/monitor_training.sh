#!/bin/bash

# Monitor DeepSpeed training progress

OUTPUT_DIR="outputs/deepspeed"
LOG_DIR="${OUTPUT_DIR}/logs"
PID_FILE="${OUTPUT_DIR}/training.pid"

# Function to display status
show_status() {
    echo "=== DeepSpeed Training Monitor ==="
    echo "Time: $(date)"
    echo "================================="
    
    # Check if PID file exists
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Training PID: $PID"
        
        # Check if process is running
        if ps -p $PID > /dev/null; then
            echo "Status: RUNNING ✓"
            
            # Show CPU and memory usage
            ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd | tail -n 1
        else
            echo "Status: NOT RUNNING ✗"
            echo "Process may have completed or crashed. Check logs for details."
        fi
    else
        echo "No training PID file found. Training may not be running."
    fi
    
    echo ""
    
    # Find latest log file
    LATEST_LOG=$(ls -t "${LOG_DIR}"/training_*.log 2>/dev/null | head -n 1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "Latest log file: $LATEST_LOG"
        echo ""
        
        # Extract training progress
        echo "=== Training Progress ==="
        
        # Show last epoch info
        LAST_EPOCH=$(grep -E "Epoch [0-9]+ \| Train Loss:" "$LATEST_LOG" | tail -n 1)
        if [ -n "$LAST_EPOCH" ]; then
            echo "Last completed: $LAST_EPOCH"
        fi
        
        # Show current batch progress
        CURRENT_BATCH=$(grep -E "Epoch [0-9]+ \| Batch" "$LATEST_LOG" | tail -n 1)
        if [ -n "$CURRENT_BATCH" ]; then
            echo "Current batch: $CURRENT_BATCH"
        fi
        
        # Show validation info
        LAST_VAL=$(grep -E "Epoch [0-9]+ \| Validation Loss:" "$LATEST_LOG" | tail -n 1)
        if [ -n "$LAST_VAL" ]; then
            echo "Last validation: $LAST_VAL"
        fi
        
        # Show best model info
        BEST_MODEL=$(grep "Saved best model" "$LATEST_LOG" | tail -n 1)
        if [ -n "$BEST_MODEL" ]; then
            echo "$BEST_MODEL"
        fi
        
        echo ""
        
        # Show any recent errors
        ERRORS=$(grep -i "error" "$LATEST_LOG" | tail -n 5)
        if [ -n "$ERRORS" ]; then
            echo "=== Recent Errors ==="
            echo "$ERRORS"
            echo ""
        fi
        
        # Show GPU memory usage from logs
        echo "=== GPU Memory Usage ==="
        grep "memory_allocated_gb" "$LATEST_LOG" | tail -n 1
        
    else
        echo "No log files found in ${LOG_DIR}"
    fi
    
    echo ""
    echo "=== Saved Checkpoints ==="
    ls -lh "${OUTPUT_DIR}"/checkpoint_epoch_* 2>/dev/null | tail -n 5
    
    echo ""
    echo "To view live logs: tail -f $LATEST_LOG"
    echo "To stop training: kill $(cat $PID_FILE 2>/dev/null)"
}

# Check command line arguments
case "$1" in
    "tail"|"-f"|"--follow")
        # Follow log file
        LATEST_LOG=$(ls -t "${LOG_DIR}"/training_*.log 2>/dev/null | head -n 1)
        if [ -n "$LATEST_LOG" ]; then
            tail -f "$LATEST_LOG"
        else
            echo "No log file found"
        fi
        ;;
    "gpu"|"--gpu")
        # Show GPU usage
        watch -n 1 nvidia-smi
        ;;
    *)
        # Show status
        if [ "$1" = "watch" ] || [ "$1" = "-w" ]; then
            # Continuous monitoring
            watch -n 5 "bash $0"
        else
            show_status
        fi
        ;;
esac 