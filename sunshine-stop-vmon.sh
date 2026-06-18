#!/usr/bin/env bash

# Disable virtual display first (before killing the process, while KDE still knows about it)
kscreen-doctor "output.Virtual-sunshine-vmon.disable" 2>/dev/null || true
sleep 1

# Kill virtual display process
if [ -f /tmp/sunshine-vmon.pid ]; then
  kill "$(cat /tmp/sunshine-vmon.pid)" 2>/dev/null || true
  rm -f /tmp/sunshine-vmon.pid
fi

# Re-enable physical monitors only
kscreen-doctor --json | jq -r '.outputs[] | select(.name != "Virtual-sunshine-vmon") | .name' | while read -r output; do
  kscreen-doctor "output.${output}.enable"
  sleep 0.5
done

# Restore Sunshine output to physical display
CONF="${HOME}/.config/sunshine/sunshine.conf"
sed -i '/^output_name/d' "$CONF"
echo "output_name = DP-1" >> "$CONF"
