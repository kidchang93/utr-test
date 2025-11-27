#!/bin/bash

# Default values
TRAIN_RATIO=0.8

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train-ratio) TRAIN_RATIO="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Triggering S3 Incremental Update (Train Ratio: $TRAIN_RATIO)..."

response=$(curl -s -X POST "http://localhost:8000/v1/prepares/dataset/s3/update" \
     -H "Content-Type: application/json" \
     -d "{
           \"train_ratio\": $TRAIN_RATIO,
           \"raw_prefix\": \"foods\",
           \"train_prefix\": \"train\",
           \"val_prefix\": \"val\"
         }")

echo "Response:"
echo "$response"
