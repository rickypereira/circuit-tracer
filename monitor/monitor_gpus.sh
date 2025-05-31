#!/bin/bash

# This is the file where the logs will be stored.
# If a parameter is provided, use it; otherwise, use the default.
LOG_FILE="${1:-gpu_log.csv}"

# Set the polling interval in seconds.
# If a second parameter is provided, use it; otherwise, use the default.
INTERVAL="${2:-5}"

# Write the CSV header to the log file.
# This command is run once before the loop starts.
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader \
  -f "$LOG_FILE" || { echo "Error writing header to $LOG_FILE. Exiting."; exit 1; }

echo "Starting GPU monitoring. Logs will be saved to $LOG_FILE every $INTERVAL seconds."

# Start the monitoring loop.
# It will append data to the file without repeating the header.
while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader \
    >> "$LOG_FILE" || { echo "Error querying GPU data. Continuing to loop..."; }
  sleep "$INTERVAL"
done
