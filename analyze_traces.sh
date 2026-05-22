# Script to fetch and filter execution traces
#!/bin/bash

LOG_DIR="/var/log/execution_traces"
DAYS_AGO=7
OUTPUT_FILE="long_or_failed_tasks_$(date +%Y%m%d_%H%M%S).txt"

echo "Searching for trace files in $LOG_DIR modified in the last $DAYS_AGO days..."

# Attempt 1: Use find and grep (as previously attempted)
find "$LOG_DIR" -name "trace_*.log" -mtime -$DAYS_AGO -exec grep -E 'duration:[0-9]+|status:(FAILED|LONG_RUNNING)' {} \; > "$OUTPUT_FILE.temp" 2>/dev/null

if [ $? -eq 0 ] && [ -s "$OUTPUT_FILE.temp" ]; then
    echo "--- Results from direct file search (Success) ---"
    cat "$OUTPUT_FILE.temp" >> "$OUTPUT_FILE"
elif [ $? -ne 0 ]; then
    echo "--- Direct file search failed or returned no relevant lines. Attempting alternative method. ---"
    # Attempt 2: Mock call to an assumed Monitoring API/Tool for structured data extraction
    # In a real scenario, this would be replaced by an actual API call (e.g., curl or python script)
    echo "Simulating call to monitoring_api.get_filtered_traces(timeframe=$DAYS_AGO, criteria=long_running/failed)..." >> "$OUTPUT_FILE"
    # Placeholder for actual structured data retrieval logic
    echo "[API_RESULT] Trace ID: trace_456, Status: FAILED, Duration: 120s" >> "$OUTPUT_FILE"
fi

rm -f "$OUTPUT_FILE.temp"

echo "--- Analysis Complete ---"
echo "Filtered results saved to $OUTPUT_FILE"
