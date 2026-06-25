#!/bin/bash
# start_server.sh — Start the AgentSight FastAPI backend
# Usage: ./start_server.sh

set -e
cd "/home/minato/Documents/Agentic Ai Project/agentsight"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " AgentSight API Server"
echo " API docs: http://localhost:8000/docs"
echo " Dashboard: run  cd dashboard_react && npm run dev"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source venv/bin/activate
venv/bin/uvicorn product.api_server:app --reload --port 8000 --host 0.0.0.0
