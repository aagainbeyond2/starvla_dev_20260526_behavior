#!/bin/bash

# Utility function to find an available port
# Requires: used_ports array to be declared in the sourcing script before sourcing this file
# Usage: port=$(find_available_port $base_port)
find_available_port() {
  local base_port=$1
  local port=$base_port

  while [[ " ${used_ports[@]} " =~ " ${port} " ]]; do
    port=$((port + 1))
  done

  if command -v netstat >/dev/null 2>&1; then
    while netstat -tuln 2>/dev/null | grep -q ":$port "; do
      port=$((port + 1))
    done
  elif command -v ss >/dev/null 2>&1; then
    while ss -tuln 2>/dev/null | grep -q ":$port "; do
      port=$((port + 1))
    done
  elif command -v lsof >/dev/null 2>&1; then
    while lsof -i :$port >/dev/null 2>&1; do
      port=$((port + 1))
    done
  else
    echo "Warning: No port checking tools available, using port ${port}" >&2
  fi

  used_ports+=($port)
  echo $port
}

print_server_log_tail() {
  local log_path="${1:-}"
  if [[ -n "${log_path}" && -f "${log_path}" ]]; then
    echo "----- server log tail: ${log_path} -----" >&2
    tail -n 40 "${log_path}" >&2 || true
    echo "----- end server log tail -----" >&2
  fi
}

check_server_health() {
  local health_url="$1"

  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 1 "${health_url}" >/dev/null 2>&1
    return $?
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -T 1 -O - "${health_url}" >/dev/null 2>&1
    return $?
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY >/dev/null 2>&1
import sys
import urllib.request
try:
    with urllib.request.urlopen("${health_url}", timeout=1) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    return $?
  fi
  if command -v python >/dev/null 2>&1; then
    python - <<PY >/dev/null 2>&1
import sys
import urllib.request
try:
    with urllib.request.urlopen("${health_url}", timeout=1) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    return $?
  fi

  return 2
}

wait_for_server() {
  local port=$1
  local pid="${2:-}"
  local log_path="${3:-}"
  local max_attempts=30
  local attempt=0
  local health_url="http://127.0.0.1:${port}/healthz"

  echo "Waiting for server on port ${port} to be ready..." >&2
  while [ $attempt -lt $max_attempts ]; do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "Service process ${pid} exited before becoming ready on port ${port}" >&2
      print_server_log_tail "${log_path}"
      return 1
    fi

    check_server_health "${health_url}"
    local health_status=$?
    if [[ ${health_status} -eq 0 ]]; then
      if [[ -z "${pid}" ]] || kill -0 "${pid}" 2>/dev/null; then
        echo "Server on port ${port} is ready" >&2
        return 0
      fi
    elif [[ ${health_status} -eq 2 ]]; then
      echo "No supported HTTP client available to probe ${health_url}" >&2
      print_server_log_tail "${log_path}"
      return 1
    fi

    sleep 2
    attempt=$((attempt + 1))
  done

  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
    echo "Service process ${pid} exited before health check succeeded on port ${port}" >&2
  else
    echo "Server on port ${port} failed healthz after $((max_attempts * 2)) seconds" >&2
  fi
  print_server_log_tail "${log_path}"
  return 1
}
