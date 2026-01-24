#!/bin/bash
PROJECT_ID="trading-algo-3"
echo "Checking billing status for $PROJECT_ID..."
gcloud beta billing projects describe $PROJECT_ID
