#!/usr/bin/env bash

WIDTH="${SUNSHINE_CLIENT_WIDTH:-1920}"
HEIGHT="${SUNSHINE_CLIENT_HEIGHT:-1080}"
FPS="${SUNSHINE_CLIENT_FPS%.*}"
FPS="${FPS:-60}"
FPS_MHZ=$(( FPS * 1000 ))
RES="${WIDTH}x${HEIGHT}"

NAME="sunshine-vmon"

# Create virtual display at client resolution
krfb-virtualmonitor --resolution "$RES" --name "$NAME" --password "sunshinepass" --port 5905 &
echo $! > /tmp/sunshine-vmon.pid

# Wait for KDE to register the new display
sleep 3

# Add custom mode support for the correct frame rate
kscreen-doctor output.Virtual-${NAME}.addCustomMode.${WIDTH}.${HEIGHT}.${FPS_MHZ}.full

# addCustomMode only registers the mode; it does not switch to it. Find the
# matching mode id and apply it so we don't have to pick it manually in the
# KDE display settings. Retry a few times in case KDE hasn't published the
# new mode list yet.
for _ in 1 2 3 4 5; do
  MODE_ID=$(kscreen-doctor --json | jq -r \
    --arg v "Virtual-${NAME}" --argjson w "$WIDTH" --argjson h "$HEIGHT" --argjson fps "$FPS" '
      .outputs[] | select(.name == $v) | .modes[]
      | select(.size.width == $w and .size.height == $h
               and ((.refreshRate - $fps) < 1) and ((.refreshRate - $fps) > -1))
      | .id' | head -n1)
  if [ -n "$MODE_ID" ]; then
    kscreen-doctor "output.Virtual-${NAME}.mode.${MODE_ID}"
    break
  fi
  sleep 0.5
done

# Disables all OTHER monitors
kscreen-doctor --json |
jq -r --arg vname "Virtual-$NAME" '
  .outputs[]
  | select(.name != $vname)
  | .name
' |
while read -r output; do
  kscreen-doctor "output.${output}.disable"
  sleep 0.5
done

# Tell Sunshine to capture this display
CONF="${HOME}/.config/sunshine/sunshine.conf"
sed -i '/^output_name/d' "$CONF"
echo "output_name = Virtual-${NAME}" >> "$CONF"
