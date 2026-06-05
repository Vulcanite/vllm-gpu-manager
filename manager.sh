#!/bin/bash

PORT=${2:-9090}

case "$1" in
    start)
        if pgrep -x "vllm-manager" > /dev/null; then
            echo "Error: vllm-manager is already running."
            exit 1
        fi
        echo "Starting vllm-manager on port $PORT in the background..."
        nohup ./vllm-manager --dashboard --port "$PORT" > vllm-manager.log 2>&1 &
        echo "Done! Server is running. Logs: tail -f vllm-manager.log"
        ;;
    stop)
        if pkill -x "vllm-manager"; then
            echo "vllm-manager background process has been stopped."
        else
            echo "vllm-manager is not currently running."
        fi
        ;;
    status)
        if pgrep -x "vllm-manager" > /dev/null; then
            echo "vllm-manager is RUNNING (PID: $(pgrep -x vllm-manager))"
        else
            echo "vllm-manager is STOPPED."
        fi
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start "$PORT"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [port]"
        echo "  [port] is optional and defaults to 9090 when starting."
        exit 1
        ;;
esac
