#!/bin/bash

# Kill distributed training runs
# Usage: ./kill_training.sh [option]

function show_running_trainings {
    echo "=== Currently Running Training Processes ==="
    echo ""
    
    local count=0
    if [ -d "logs" ]; then
        for pid_file in logs/dist_train_*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                timestamp=$(basename "$pid_file" | sed 's/dist_train_\(.*\)\.pid/\1/')
                log_file="logs/dist_train_${timestamp}.log"
                
                if ps -p $pid > /dev/null 2>&1; then
                    count=$((count + 1))
                    echo "[$count] Training $timestamp (PID: $pid)"
                    if [ -f "$log_file" ]; then
                        # Get last epoch info from log
                        last_epoch=$(grep -o "Epoch [0-9]*/[0-9]*" "$log_file" | tail -1)
                        if [ ! -z "$last_epoch" ]; then
                            echo "    Progress: $last_epoch"
                        fi
                        # Get start time
                        start_time=$(head -1 "$log_file" | grep -o "[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}" | head -1)
                        if [ ! -z "$start_time" ]; then
                            echo "    Started: $start_time"
                        fi
                    fi
                    echo ""
                fi
            fi
        done
    fi
    
    if [ $count -eq 0 ]; then
        echo "No running training processes found."
        exit 0
    fi
    
    return $count
}

function kill_by_pid {
    local pid=$1
    if ps -p $pid > /dev/null 2>&1; then
        echo "Killing training process (PID: $pid)..."
        # Kill the torchrun process and all its children
        pkill -P $pid
        kill $pid
        sleep 2
        
        # Force kill if still running
        if ps -p $pid > /dev/null 2>&1; then
            echo "Process still running, force killing..."
            kill -9 $pid
            pkill -9 -P $pid
        fi
        
        echo "Training process $pid has been terminated."
    else
        echo "Process $pid is not running."
    fi
}

function kill_latest {
    echo "Finding latest training run..."
    latest_pid_file=$(ls -t logs/dist_train_*.pid 2>/dev/null | head -1)
    
    if [ -f "$latest_pid_file" ]; then
        pid=$(cat "$latest_pid_file")
        timestamp=$(basename "$latest_pid_file" | sed 's/dist_train_\(.*\)\.pid/\1/')
        echo "Latest training: $timestamp (PID: $pid)"
        kill_by_pid $pid
    else
        echo "No training runs found."
    fi
}

function kill_all {
    echo "Killing ALL training processes..."
    
    # Kill all torchrun processes related to dist_train.py
    pkill -f "torchrun.*dist_train.py"
    
    # Also check PID files
    if [ -d "logs" ]; then
        for pid_file in logs/dist_train_*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null 2>&1; then
                    kill_by_pid $pid
                fi
            fi
        done
    fi
    
    echo "All training processes have been terminated."
}

function interactive_kill {
    show_running_trainings
    local num_processes=$?
    
    if [ $num_processes -eq 0 ]; then
        exit 0
    fi
    
    echo -n "Enter the number of the training to kill (or 'all' for all): "
    read choice
    
    if [ "$choice" = "all" ]; then
        kill_all
    elif [[ "$choice" =~ ^[0-9]+$ ]]; then
        # Find the nth running process
        local count=0
        if [ -d "logs" ]; then
            for pid_file in logs/dist_train_*.pid; do
                if [ -f "$pid_file" ]; then
                    pid=$(cat "$pid_file")
                    if ps -p $pid > /dev/null 2>&1; then
                        count=$((count + 1))
                        if [ $count -eq $choice ]; then
                            timestamp=$(basename "$pid_file" | sed 's/dist_train_\(.*\)\.pid/\1/')
                            echo "Killing training $timestamp (PID: $pid)..."
                            kill_by_pid $pid
                            exit 0
                        fi
                    fi
                fi
            done
        fi
        echo "Invalid selection."
    else
        echo "Invalid input. Exiting."
    fi
}

# Check if nvidia-smi is available to show GPU memory release
function show_gpu_memory {
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "=== GPU Memory Status ==="
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %d: %5.0f / %5.0f MB\n", $1, $2, $3}'
    fi
}

# Main logic
case "$1" in
    latest)
        kill_latest
        show_gpu_memory
        ;;
    all)
        kill_all
        show_gpu_memory
        ;;
    -h|--help)
        echo "Kill distributed training runs"
        echo ""
        echo "Usage: ./kill_training.sh [option]"
        echo ""
        echo "Options:"
        echo "  (no option)  - Interactive mode, shows running processes and prompts for selection"
        echo "  latest       - Kill the most recent training run"
        echo "  all          - Kill all training runs"
        echo "  -h, --help   - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./kill_training.sh          # Interactive selection"
        echo "  ./kill_training.sh latest   # Kill most recent training"
        echo "  ./kill_training.sh all      # Kill all trainings"
        ;;
    *)
        interactive_kill
        show_gpu_memory
        ;;
esac 