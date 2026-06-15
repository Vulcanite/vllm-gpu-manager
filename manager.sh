#!/bin/bash

PORT=9090
PASSWORD=""
PROXY=""

for ARG in "${@:2}"; do
    case "$ARG" in
        --proxy)
            PROXY=1
            ;;
        *)
            if [ "$PORT" = "9090" ] && [[ "$ARG" =~ ^[0-9]+$ ]]; then
                PORT=$ARG
            elif [ -z "$PASSWORD" ]; then
                PASSWORD=$ARG
            else
                echo "Error: unexpected argument: $ARG"
                exit 1
            fi
            ;;
    esac
done

case "$1" in
    start)
        if pgrep -x "vllm-manager" > /dev/null; then
            echo "Error: vllm-manager is already running."
            exit 1
        fi
        
        if [ -n "$PASSWORD" ]; then
            export VLLM_MANAGER_PASSWORD="$PASSWORD"
            echo "Authentication enabled with provided password."
        elif [ -n "$VLLM_MANAGER_PASSWORD" ]; then
            echo "Authentication enabled from existing environment variable."
        fi

        ARGS=(--dashboard --port "$PORT")
        if [ -n "$PROXY" ] || [ "$VLLM_MANAGER_PROXY" = "1" ] || [ "$VLLM_MANAGER_PROXY" = "true" ]; then
            ARGS+=(--proxy)
            echo "Proxy gateway enabled."
        fi

        echo "Starting vllm-manager on port $PORT in the background..."
        nohup ./vllm-manager "${ARGS[@]}" > vllm-manager.log 2>&1 &
        echo "Done! Server is running at https://localhost:$PORT"
        echo "Logs: tail -f vllm-manager.log"
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
        $0 start "$PORT" ${PASSWORD:+"$PASSWORD"} ${PROXY:+--proxy}
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [port] [password] [--proxy]"
        echo "  [port] defaults to 9090."
        echo "  [password] is optional and secures the dashboard."
        echo "  --proxy enables the /v1/ OpenAI-compatible proxy gateway."
        exit 1
        ;;
esac
