#!/bin/bash

STATE_FILE=".vllm-manager.state"
PORT=9090
PORT_SET=0
HOST=""
HOST_SET=0
PASSWORD=""
NO_PROXY=""
NO_PROXY_SET=0

ARGS_INPUT=("${@:2}")
for ((i = 0; i < ${#ARGS_INPUT[@]}; i++)); do
    ARG="${ARGS_INPUT[$i]}"
    case "$ARG" in
        --host)
            if [ $((i + 1)) -ge ${#ARGS_INPUT[@]} ]; then
                echo "Error: --host requires an IP address."
                exit 1
            fi
            HOST="${ARGS_INPUT[$((i + 1))]}"
            HOST_SET=1
            i=$((i + 1))
            ;;
        --host=*)
            HOST="${ARG#--host=}"
            if [ -z "$HOST" ]; then
                echo "Error: --host requires an IP address."
                exit 1
            fi
            HOST_SET=1
            ;;
        --no-proxy)
            NO_PROXY=1
            NO_PROXY_SET=1
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
    local saved_host=""
    local saved_no_proxy=""
    local saved_password=""
    while IFS='=' read -r key value; do
        case "$key" in
            PORT)
                if [[ "$value" =~ ^[0-9]+$ ]]; then
                    saved_port="$value"
                fi
                ;;
            HOST)
                saved_host="$value"
                ;;
            NO_PROXY)
                if [ "$value" = "1" ]; then
                    saved_no_proxy=1
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
    if [ "$HOST_SET" -eq 0 ] && [ -n "$saved_host" ]; then
        HOST="$saved_host"
    fi
    if [ "$NO_PROXY_SET" -eq 0 ]; then
        NO_PROXY="$saved_no_proxy"
    fi
    if [ -z "$PASSWORD" ] && [ -n "$saved_password" ]; then
        PASSWORD="$saved_password"
    fi
}

save_state() {
    local no_proxy_value="$1"
    local password_value="$2"
    {
        echo "PORT=$PORT"
        echo "HOST=$HOST"
        echo "NO_PROXY=$no_proxy_value"
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
        if [ -n "$HOST" ]; then
            ARGS+=(--host "$HOST")
        fi
        NO_PROXY_ENABLED=""
        if [ -n "$NO_PROXY" ] || [ "$VLLM_MANAGER_NO_PROXY" = "1" ] || [ "$VLLM_MANAGER_NO_PROXY" = "true" ]; then
            ARGS+=(--no-proxy)
            NO_PROXY_ENABLED=1
            echo "Proxy gateway disabled."
        fi

        save_state "${NO_PROXY_ENABLED:-0}" "$ACTIVE_PASSWORD"
        DISPLAY_HOST="${HOST:-127.0.0.1}"
        echo "Starting vllm-manager on $DISPLAY_HOST:$PORT in the background..."
        nohup ./vllm-manager "${ARGS[@]}" > vllm-manager.log 2>&1 &
        echo "Done! Server is running at https://$DISPLAY_HOST:$PORT"
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
        RESTART_ARGS=(start "$PORT")
        if [ -n "$HOST" ]; then
            RESTART_ARGS+=(--host "$HOST")
        fi
        if [ -n "$PASSWORD" ]; then
            RESTART_ARGS+=("$PASSWORD")
        fi
        if [ -n "$NO_PROXY" ]; then
            RESTART_ARGS+=(--no-proxy)
        fi
        "$0" "${RESTART_ARGS[@]}"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [port] [password] [--host <ip>] [--no-proxy]"
        echo "  [port] defaults to 9090."
        echo "  [password] is optional and secures the dashboard."
        echo "  --host sets the dashboard bind IP, defaulting to the binary default."
        echo "  --no-proxy disables the /v1/ OpenAI-compatible proxy gateway."
        exit 1
        ;;
esac
