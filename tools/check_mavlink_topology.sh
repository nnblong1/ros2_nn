#!/usr/bin/env bash
# Read-only MAVLink/XRCE topology audit for the Raspberry Pi.
# It does not kill processes, change PX4 params, or start any service.

set -u

MAVLINK_PORT_RE='14540|14550|14555|14557|14580'
PROC_RE='[m]avlink|[m]avros|[M]AVSDK|[M]AVProxy|[r]outer|[Q]GroundControl|[M]icroXRCEAgent'
SERIAL_DEVICES=(/dev/ttyACM0 /dev/serial0 /dev/ttyAMA0 /dev/ttyS0)

section() {
  printf '\n== %s ==\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

run_or_warn() {
  local title="$1"
  shift
  section "$title"
  if "$@"; then
    return 0
  fi
  printf 'WARN: command failed:'
  printf ' %q' "$@"
  printf '\n'
  return 1
}

capture_lsof() {
  if have sudo && sudo -n true 2>/dev/null; then
    sudo lsof -l "${SERIAL_DEVICES[@]}" 2>/dev/null
  elif have lsof; then
    lsof -l "${SERIAL_DEVICES[@]}" 2>/dev/null
  else
    return 127
  fi
}

local_ipv4_addresses() {
  if have hostname; then
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true
  elif have ip; then
    ip -o -4 addr show 2>/dev/null | awk '{split($4, a, "/"); print a[1]}'
  fi
}

section "Serial aliases"
if have ls; then
  ls -l /dev/serial0 /dev/serial1 2>/dev/null || true
else
  printf 'WARN: ls not found\n'
fi

if have readlink; then
  serial0_target="$(readlink -f /dev/serial0 2>/dev/null || true)"
  serial1_target="$(readlink -f /dev/serial1 2>/dev/null || true)"
  printf 'serial0 target: %s\n' "${serial0_target:-not found}"
  printf 'serial1 target: %s\n' "${serial1_target:-not found}"
fi

section "Serial device owners"
lsof_output="$(capture_lsof || true)"
if [ -n "$lsof_output" ]; then
  printf '%s\n' "$lsof_output"
else
  printf 'No serial owners found, or lsof requires sudo password.\n'
fi

section "MAVLink-related UDP listeners"
if have ss; then
  ss -lunp 2>/dev/null | grep -E "$MAVLINK_PORT_RE" || printf 'No listeners on common MAVLink UDP ports.\n'
else
  printf 'WARN: ss not found\n'
fi

section "MAVLink/XRCE-related processes"
if have pgrep; then
  raw_proc_output="$(pgrep -af "$PROC_RE" || true)"
  proc_output="$(printf '%s\n' "$raw_proc_output" | awk -v self="$$" '$1 != self && $0 !~ /check_mavlink_topology[.]sh/')"
  if [ -n "$proc_output" ]; then
    printf '%s\n' "$proc_output"
  else
    printf 'No matching MAVLink/XRCE process found.\n'
  fi
elif have ps; then
  ps -ef | awk -v self="$$" -v proc_re="$PROC_RE" '$2 != self && $0 !~ /check_mavlink_topology[.]sh/ && $0 ~ proc_re'
else
  printf 'WARN: ps not found\n'
fi

section "Warnings"
warned=0
local_ips="$(local_ipv4_addresses)"

if printf '%s\n' "$lsof_output" | grep -qE 'MicroXRCE.*/dev/ttyACM[0-9]'; then
  printf 'WARN: MicroXRCEAgent is using /dev/ttyACM*. PX4 USB CDC must be reserved for mavlink-routerd/QGC.\n'
  warned=1
fi

if printf '%s\n' "$lsof_output" | grep -qE 'MicroXRCE.*/dev/ttyAMA0'; then
  printf 'WARN: MicroXRCEAgent is using /dev/ttyAMA0. For this setup TELEM2 should use /dev/serial0.\n'
  warned=1
fi

if have pgrep; then
  xrce_count="$(pgrep -fc 'MicroXRCEAgent' || true)"
elif have ps; then
  xrce_count="$(ps -ef | awk '/[M]icroXRCEAgent/{count++} END {print count + 0}')"
else
  xrce_count=0
fi

if [ "$xrce_count" -gt 1 ]; then
  printf 'WARN: more than one MicroXRCEAgent process is running.\n'
  warned=1
fi

if { have pgrep && pgrep -f 'QGroundControl' >/dev/null; } || \
   { have ps && ps -ef | awk '/[Q]GroundControl/{found=1} END {exit !found}'; }; then
  printf 'WARN: QGroundControl appears to be running on this Pi. Keep QGC only on the laptop during loop tests.\n'
  warned=1
fi

if { have pgrep && pgrep -f 'mavros|MAVSDK|MAVProxy' >/dev/null; } || \
   { have ps && ps -ef | awk '/[m]avros|[M]AVSDK|[M]AVProxy/{found=1} END {exit !found}'; }; then
  printf 'WARN: MAVROS/MAVSDK/MAVProxy appears to be running. Stop it while isolating the USB CDC MAVLink route.\n'
  warned=1
fi

if [ -n "${proc_output:-}" ] && [ -n "$local_ips" ]; then
  mavlink_router_endpoints="$(
    printf '%s\n' "$proc_output" | awk '
      /mavlink-routerd/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "-e" && (i + 1) <= NF) {
            split($(i + 1), endpoint, ":")
            print endpoint[1]
          }
        }
      }
    '
  )"
  for endpoint_ip in $mavlink_router_endpoints; do
    if printf '%s\n' "$local_ips" | grep -Fxq "$endpoint_ip"; then
      printf 'WARN: mavlink-routerd endpoint %s is a local Pi IP. Use the laptop QGC IP instead, otherwise UDP can loop back into mavlink-routerd.\n' "$endpoint_ip"
      warned=1
    fi
  done
fi

if [ "$warned" -eq 0 ]; then
  printf 'No obvious Pi-side topology conflict found.\n'
fi

section "PX4 NSH checklist"
cat <<'EOF'
Run on PX4 NSH and reboot after saving:

param show MAV_*_FORWARD
param set MAV_0_FORWARD 0
param set MAV_1_FORWARD 0
param set MAV_2_FORWARD 0
param set MAV_3_FORWARD 0
param set MAV_HB_FORW_EN 0

param show MAV_*_CONFIG
param show XRCE_DDS_CFG

# If any MAV_N_CONFIG is TELEM2/102, disable that MAVLink instance.
# Example: if MAV_1_CONFIG is TELEM2/102:
# param set MAV_1_CONFIG 0
param set XRCE_DDS_CFG 102
param save
reboot

After reboot, mavlink status on USB CDC must show Forwarding: Off.
EOF
