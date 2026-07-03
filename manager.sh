#!/bin/bash

STATE_FILE=".vllm-manager.state"
PORT=9090
PORT_SET=0
PASSWORD=""
PROXY=""
PROXY_SET=0

for ARG in "${@:2}"; do
    case "$ARG" in
        --proxy)
            PROXY=1
            PROXY_SET=1
            ;;
        *)
            if [ "$PORT_SET" -eq 0 ] && [[ "$ARG" =~ ^[0-9]+$ ]]; then
                PORT=$ARG
                PORT_SET=1
            elif [ -z "$PASSWORD" ]; then
                PASSWORD=$ARG
            else
                echo "Error: unexpected argument: $ARG"
                exit 1
            fi
            ;;
    esac
done

load_state_for_restart() {
    [ "$1" = "restart" ] || return 0
    [ -f "$STATE_FILE" ] || return 0

    local saved_port=""
    local saved_proxy=""
    local saved_password=""
    while IFS='=' read -r key value; do
        case "$key" in
            PORT)
                if [[ "$value" =~ ^[0-9]+$ ]]; then
                    saved_port="$value"
                fi
                ;;
            PROXY)
                if [ "$value" = "1" ]; then
                    saved_proxy=1
                fi
                ;;
            PASSWORD)
                saved_password="$value"
                ;;
        esac
    done < "$STATE_FILE"

    if [ "$PORT_SET" -eq 0 ] && [ -n "$saved_port" ]; then
        PORT="$saved_port"
    fi
    if [ "$PROXY_SET" -eq 0 ]; then
        PROXY="$saved_proxy"
    fi
    if [ -z "$PASSWORD" ] && [ -n "$saved_password" ]; then
        PASSWORD="$saved_password"
    fi
}

save_state() {
    local proxy_value="$1"
    local password_value="$2"
    {
        echo "PORT=$PORT"
        echo "PROXY=$proxy_value"
        echo "PASSWORD=$password_value"
    } > "$STATE_FILE"
    chmod 600 "$STATE_FILE"
}

load_state_for_restart "$1"

case "$1" in
    start)
        if pgrep -x "vllm-manager" > /dev/null; then
            echo "Error: vllm-manager is already running."
            exit 1
        fi
        
        ACTIVE_PASSWORD="$PASSWORD"
        if [ -n "$ACTIVE_PASSWORD" ]; then
            export VLLM_MANAGER_PASSWORD="$PASSWORD"
            echo "Authentication enabled with provided password."
        elif [ -n "$VLLM_MANAGER_PASSWORD" ]; then
            ACTIVE_PASSWORD="$VLLM_MANAGER_PASSWORD"
            echo "Authentication enabled from existing environment variable."
        fi

        ARGS=(--dashboard --port "$PORT")
        PROXY_ENABLED=""
        if [ -n "$PROXY" ] || [ "$VLLM_MANAGER_PROXY" = "1" ] || [ "$VLLM_MANAGER_PROXY" = "true" ]; then
            ARGS+=(--proxy)
            PROXY_ENABLED=1
            echo "Proxy gateway enabled."
        fi

        save_state "${PROXY_ENABLED:-0}" "$ACTIVE_PASSWORD"
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
        "$0" stop
        for _ in {1..15}; do
            if ! pgrep -x "vllm-manager" > /dev/null; then
                break
            fi
            sleep 1
        done
        if pgrep -x "vllm-manager" > /dev/null; then
            echo "Error: vllm-manager did not stop in time; restart aborted."
            exit 1
        fi
        "$0" start "$PORT" ${PASSWORD:+"$PASSWORD"} ${PROXY:+--proxy}
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [port] [password] [--proxy]"
        echo "  [port] defaults to 9090."
        echo "  [password] is optional and secures the dashboard."
        echo "  --proxy enables the /v1/ OpenAI-compatible proxy gateway."
        exit 1
        ;;
esac
