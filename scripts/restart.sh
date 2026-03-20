#!/usr/bin/env bash
# Quick restart of all ReviewMind app services (bot, API, worker, beat).
# Docker infra (postgres, redis, qdrant) is left running.
# Usage: source .env && bash scripts/restart.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "⏹  Stopping app processes..."
pkill -f "reviewmind.bot.main" 2>/dev/null || true
pkill -f "uvicorn reviewmind.main" 2>/dev/null || true
pkill -f "celery.*reviewmind.workers" 2>/dev/null || true
sleep 2

echo "▶  Starting bot..."
set -a && source .env && set +a
nohup uv run python -m reviewmind.bot.main > /tmp/reviewmind-bot.log 2>&1 &

echo "▶  Starting API..."
nohup uv run uvicorn reviewmind.main:app --host 0.0.0.0 --port 8000 > /tmp/reviewmind-api.log 2>&1 &

echo "▶  Starting Celery worker..."
nohup uv run celery -A reviewmind.workers.celery_app worker --loglevel=info > /tmp/reviewmind-worker.log 2>&1 &

echo "▶  Starting Celery beat..."
nohup uv run celery -A reviewmind.workers.celery_app beat --loglevel=info > /tmp/reviewmind-beat.log 2>&1 &

sleep 3
echo ""
echo "✅ All services restarted. Logs:"
echo "   Bot:    tail -f /tmp/reviewmind-bot.log"
echo "   API:    tail -f /tmp/reviewmind-api.log"
echo "   Worker: tail -f /tmp/reviewmind-worker.log"
echo "   Beat:   tail -f /tmp/reviewmind-beat.log"
echo "   All:    tail -f /tmp/reviewmind-*.log"
