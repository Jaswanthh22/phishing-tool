#!/usr/bin/env bash
echo "Stopping local PhishGuard API..."
pkill -f "uvicorn" || true
echo "Stopped."
