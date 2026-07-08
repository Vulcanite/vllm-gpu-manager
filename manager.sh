#!/bin/bash

STATE_FILE=".vllm-manager.state"
PORT=9090
PORT_SET=0
HOST=""
HOST_SET=0
PASSWORD=""
NO_PROXY=""
NO_PROXY_SET=0
HTTPS=""
HTTPS_SET=0
MONITORING=""
MONITORING_SET=0
MONITORING_TARGETS_FILE=""
MONITORING_TARGETS_FILE_SET=0

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
        --https)
            HTTPS=1
            HTTPS_SET=1
            ;;
        --monitoring)
            MONITORING=1
            MONITORING_SET=1
            ;;
        --monitoring-targets-file)
            if [ $((i + 1)) -ge ${#ARGS_INPUT[@]} ]; then
                echo "Error: --monitoring-targets-file requires a path."
                exit 1
            fi
            MONITORING_TARGETS_FILE="${ARGS_INPUT[$((i + 1))]}"
            if [ -z "$MONITORING_TARGETS_FILE" ]; then
                echo "Error: --monitoring-targets-file requires a path."
                exit 1
            fi
            MONITORING_TARGETS_FILE_SET=1
            i=$((i + 1))
            ;;
        --monitoring-targets-file=*)
            MONITORING_TARGETS_FILE="${ARG#--monitoring-targets-file=}"
            if [ -z "$MONITORING_TARGETS_FILE" ]; then
                echo "Error: --monitoring-targets-file requires a path."
                exit 1
            fi
            MONITORING_TARGETS_FILE_SET=1
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
    local saved_https=""
    local saved_monitoring=""
    local saved_monitoring_targets_file=""
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
            HTTPS)
                if [ "$value" = "1" ]; then
                    saved_https=1
                fi
                ;;
            MONITORING)
                if [ "$value" = "1" ]; then
                    saved_monitoring=1
                fi
                ;;
            MONITORING_TARGETS_FILE)
                saved_monitoring_targets_file="$value"
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
    if [ "$HTTPS_SET" -eq 0 ]; then
        HTTPS="$saved_https"
    fi
    if [ "$MONITORING_SET" -eq 0 ]; then
        MONITORING="$saved_monitoring"
    fi
    if [ "$MONITORING_TARGETS_FILE_SET" -eq 0 ] && [ -n "$saved_monitoring_targets_file" ]; then
        MONITORING_TARGETS_FILE="$saved_monitoring_targets_file"
    fi
    if [ -z "$PASSWORD" ] && [ -n "$saved_password" ]; then
        PASSWORD="$saved_password"
    fi
}

save_state() {
    local no_proxy_value="$1"
    local https_value="$2"
    local monitoring_value="$3"
    local monitoring_targets_file_value="$4"
    local password_value="$5"
    {
        echo "PORT=$PORT"
        echo "HOST=$HOST"
        echo "NO_PROXY=$no_proxy_value"
        echo "HTTPS=$https_value"
        echo "MONITORING=$monitoring_value"
        echo "MONITORING_TARGETS_FILE=$monitoring_targets_file_value"
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
        HTTPS_ENABLED=""
        if [ -n "$HTTPS" ] || [ "$VLLM_MANAGER_HTTPS" = "1" ] || [ "$VLLM_MANAGER_HTTPS" = "true" ]; then
            ARGS+=(--https)
            HTTPS_ENABLED=1
            echo "HTTPS enabled."
        fi
        MONITORING_ENABLED=""
        ACTIVE_MONITORING_TARGETS_FILE="$MONITORING_TARGETS_FILE"
        if [ -z "$ACTIVE_MONITORING_TARGETS_FILE" ]; then
            ACTIVE_MONITORING_TARGETS_FILE="$VLLM_MANAGER_MONITORING_TARGETS_FILE"
        fi
        if [ -n "$MONITORING" ] || [ "$VLLM_MANAGER_MONITORING" = "1" ] || [ "$VLLM_MANAGER_MONITORING" = "true" ]; then
            ARGS+=(--monitoring)
            MONITORING_ENABLED=1
            echo "Monitoring target sync enabled."
        fi
        if [ -n "$ACTIVE_MONITORING_TARGETS_FILE" ]; then
            ARGS+=(--monitoring-targets-file "$ACTIVE_MONITORING_TARGETS_FILE")
            echo "Monitoring targets file: $ACTIVE_MONITORING_TARGETS_FILE"
        fi

        save_state "${NO_PROXY_ENABLED:-0}" "${HTTPS_ENABLED:-0}" "${MONITORING_ENABLED:-0}" "$ACTIVE_MONITORING_TARGETS_FILE" "$ACTIVE_PASSWORD"
        DISPLAY_HOST="${HOST:-127.0.0.1}"
        SCHEME="http"
        if [ -n "$HTTPS_ENABLED" ]; then
            SCHEME="https"
        fi
        echo "Starting vllm-manager on $DISPLAY_HOST:$PORT in the background..."
        nohup ./vllm-manager "${ARGS[@]}" > vllm-manager.log 2>&1 &
        echo "Done! Server is running at $SCHEME://$DISPLAY_HOST:$PORT"
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
        if [ -n "$HTTPS" ]; then
            RESTART_ARGS+=(--https)
        fi
        if [ -n "$MONITORING" ]; then
            RESTART_ARGS+=(--monitoring)
        fi
        if [ -n "$MONITORING_TARGETS_FILE" ]; then
            RESTART_ARGS+=(--monitoring-targets-file "$MONITORING_TARGETS_FILE")
        fi
        "$0" "${RESTART_ARGS[@]}"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [port] [password] [--host <ip>] [--no-proxy] [--https] [--monitoring] [--monitoring-targets-file <path>]"
        echo "  [port] defaults to 9090."
        echo "  [password] is optional and secures the dashboard."
        echo "  --host sets the dashboard bind IP, defaulting to the binary default."
        echo "  --no-proxy disables the /v1/ OpenAI-compatible proxy gateway."
        echo "  --https enables self-signed HTTPS on the dashboard port."
        echo "  --monitoring syncs running vLLM servers into the Prometheus file-SD targets file."
        echo "  --monitoring-targets-file sets the Prometheus file-SD path; binary default is monitoring/runtime/vllm-targets.yml."
        exit 1
        ;;
esac
