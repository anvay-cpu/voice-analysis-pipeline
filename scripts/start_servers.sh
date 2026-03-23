#!/bin/bash
# Start both the FastAPI backend and Next.js frontend
# Usage: ./scripts/start_servers.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  SPEECH COACH — Starting Servers"
echo "============================================"

# Ensure data directories exist
mkdir -p data/sessions data/uploads

# Start FastAPI backend
echo ""
echo "[1/2] Starting FastAPI backend on http://localhost:8000 ..."
conda run -n voice-pipeline uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID"

# Wait for backend to be ready
sleep 3

# Start Next.js frontend
echo ""
echo "[2/2] Starting Next.js frontend on http://localhost:3000 ..."
cd webapp && npm run dev &
FRONTEND_PID=$!
echo "  PID: $FRONTEND_PID"

echo ""
echo "============================================"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API docs: http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C to kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
