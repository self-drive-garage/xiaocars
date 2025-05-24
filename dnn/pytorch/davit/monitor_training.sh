#!/bin/bash

# Monitor distributed training status
# Usage: ./monitor_training.sh [command]

function show_help {
    echo "Distributed Training Monitor"
    echo ""
    echo "Usage: ./monitor_training.sh [command]"
    echo ""
    echo "Commands:"
    echo "  status    - Show status of all training runs"
    echo "  latest    - Show log of the latest training run"
    echo "  follow    - Follow the latest training log (tail -f)"
    echo "  gpu       - Show GPU utilization"
    echo "  kill      - Kill the latest training run"
    echo "  killall   - Kill all training runs"
    echo ""
}

function show_status {
    echo "=== Training Runs Status ==="
    echo ""
    
    if [ -d "logs" ]; then
        for pid_file in logs/dist_train_*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                timestamp=$(basename "$pid_file" | sed 's/dist_train_\(.*\)\.pid/\1/')
                log_file="logs/dist_train_${timestamp}.log"
                
                if ps -p $pid > /dev/null 2>&1; then
                    echo "✓ Training $timestamp (PID: $pid) - RUNNING"
                    if [ -f "$log_file" ]; then
                        # Get last epoch info from log
                        last_epoch=$(grep -o "Epoch [0-9]*/[0-9]*" "$log_file" | tail -1)
                        if [ ! -z "$last_epoch" ]; then
                            echo "  Last update: $last_epoch"
                        fi
                    fi
                else
                    echo "✗ Training $timestamp (PID: $pid) - STOPPED"
                fi
                echo ""
            fi
        done
    else
        echo "No training runs found."
    fi
}

function show_latest_log {
    latest_log=$(ls -t logs/dist_train_*.log 2>/dev/null | head -1)
    if [ -f "$latest_log" ]; then
        echo "Showing log: $latest_log"
        echo "Press Ctrl+C to exit"
        echo ""
        less +G "$latest_log"
    else
        echo "No log files found."
    fi
}

function follow_latest_log {
    latest_log=$(ls -t logs/dist_train_*.log 2>/dev/null | head -1)
    if [ -f "$latest_log" ]; then
        echo "Following log: $latest_log"
        echo "Press Ctrl+C to exit"
        echo ""
        tail -f "$latest_log"
    else
        echo "No log files found."
    fi
}

function show_gpu_status {
    echo "=== GPU Utilization ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %d: %s - %3d%% util, %5.0f/%5.0f MB\n", $1, $2, $3, $4, $5}'
}

function kill_latest {
    latest_pid=$(ls -t logs/dist_train_*.pid 2>/dev/null | head -1)
    if [ -f "$latest_pid" ]; then
        pid=$(cat "$latest_pid")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing training process (PID: $pid)..."
            kill $pid
            echo "Training stopped."
        else
            echo "Process $pid is not running."
        fi
    else
        echo "No training runs found."
    fi
}

function kill_all_training {
    echo "Killing all training processes..."
    pkill -f "torchrun.*dist_train.py"
    echo "All training processes stopped."
}

# Main command dispatcher
case "$1" in
    status)
        show_status
        ;;
    latest)
        show_latest_log
        ;;
    follow)
        follow_latest_log
        ;;
    gpu)
        show_gpu_status
        ;;
    kill)
        kill_latest
        ;;
    killall)
        kill_all_training
        ;;
    *)
        show_help
        ;;
esac 