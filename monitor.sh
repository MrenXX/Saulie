#!/bin/bash

# Tmux-Based Production Dashboard
# ================================
# Creates a multi-panel dashboard with:
# - Top panel: Container stats & metrics
# - Bottom panel: Live logs stream
# 
# Requirements: tmux (install with: apt install tmux / brew install tmux)

CONTAINER_NAME="eval_deploy_qwenie"
SESSION_NAME="saulie_dashboard"
API_URL="http://localhost:8000"
API_KEY="${VLLM_API_KEY:-dipshit}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    echo -e "${YELLOW}Install with:${NC}"
    echo "  Ubuntu/Debian: sudo apt install tmux"
    echo "  MacOS: brew install tmux"
    echo "  CentOS/RHEL: sudo yum install tmux"
    exit 1
fi

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${RED}Error: Container '${CONTAINER_NAME}' not found${NC}"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t ${SESSION_NAME} 2>/dev/null

echo -e "${GREEN}Starting Saulie Production Dashboard...${NC}"
echo -e "${YELLOW}Commands:${NC}"
echo "  - Ctrl+B then D: Detach (dashboard keeps running)"
echo "  - Ctrl+B then [: Scroll mode (use arrows, Q to exit)"
echo "  - Ctrl+B then :: Command mode"
echo "  - To reattach: tmux attach -t ${SESSION_NAME}"
echo "  - To kill: tmux kill-session -t ${SESSION_NAME}"
echo ""
sleep 2

# Create new tmux session
tmux new-session -d -s ${SESSION_NAME} -n "Dashboard"

# Split window horizontally (top and bottom)
tmux split-window -v -t ${SESSION_NAME}:0

# Resize panes (top 30%, bottom 70%)
tmux resize-pane -t ${SESSION_NAME}:0.0 -y 15

# Top pane: Metrics/Status (continuously updating)
tmux send-keys -t ${SESSION_NAME}:0.0 "
clear
while true; do
    clear
    echo '═════════════════════════════════════════════════════════════════'
    echo ' SAULIE PRODUCTION DASHBOARD - $(date +\"%Y-%m-%d %H:%M:%S\")'
    echo '═════════════════════════════════════════════════════════════════'
    echo ''
    
    # Container Status
    if docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'; then
        echo -e '\033[0;32m● Container: RUNNING\033[0m'
    else
        echo -e '\033[0;31m● Container: STOPPED\033[0m'
    fi
    
    # API Health
    if curl -s -o /dev/null -w '%{http_code}' --max-time 2 ${API_URL}/health 2>/dev/null | grep -q '200'; then
        echo -e '\033[0;32m● API: HEALTHY\033[0m'
    else
        echo -e '\033[0;31m● API: UNHEALTHY\033[0m'
    fi
    
    echo ''
    echo '─────────────────────────────────────────────────────────────────'
    
    # Container Stats
    docker stats ${CONTAINER_NAME} --no-stream --format 'CPU: {{.CPUPerc}}  Memory: {{.MemUsage}}  Network: {{.NetIO}}' 2>/dev/null || echo 'Stats: N/A'
    
    # GPU Stats (if available)
    if command -v nvidia-smi &> /dev/null; then
        docker exec ${CONTAINER_NAME} nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | awk -F, '{printf \"GPU: %s%%  VRAM: %sMB/%sMB  Temp: %s°C\n\", \$1, \$2, \$3, \$4}' || echo 'GPU: N/A'
    fi
    
    echo '─────────────────────────────────────────────────────────────────'
    echo 'Press Ctrl+B then D to detach | Ctrl+B then [ to scroll logs'
    echo '═════════════════════════════════════════════════════════════════'
    
    sleep 3
done
" C-m

# Bottom pane: split logs (left) + nvitop (right)
tmux select-pane -t ${SESSION_NAME}:0.1 -T "vLLM logs"
tmux split-window -h -t ${SESSION_NAME}:0.1
tmux select-pane -t ${SESSION_NAME}:0.2 -T "nvitop"
tmux send-keys -t ${SESSION_NAME}:0.1 "
docker logs ${CONTAINER_NAME} -f --tail 100 2>&1 | while IFS= read -r line; do
    # Colorize based on content
    if echo \"\$line\" | grep -qiE 'error|exception|fail|critical'; then
        echo -e \"\033[0;31m\$line\033[0m\"
    elif echo \"\$line\" | grep -qiE 'warning|warn'; then
        echo -e \"\033[1;33m\$line\033[0m\"
    elif echo \"\$line\" | grep -qiE 'success|started|ready|healthy'; then
        echo -e \"\033[0;32m\$line\033[0m\"
    elif echo \"\$line\" | grep -qiE 'tool|function|search'; then
        echo -e \"\033[0;35m\$line\033[0m\"
    else
        echo \"\$line\"
    fi
done
" C-m

tmux send-keys -t ${SESSION_NAME}:0.2 "
NVITOP=\$(command -v nvitop || true)
[[ -z \"\$NVITOP\" && -x /root/miniconda3/envs/saulgman/bin/nvitop ]] && NVITOP=/root/miniconda3/envs/saulgman/bin/nvitop
if [[ -n \"\$NVITOP\" ]]; then
  exec \"\$NVITOP\"
else
  echo '[!] nvitop not installed — pip install nvitop'
  exec bash
fi
" C-m

# Set focus to log pane
tmux select-pane -t ${SESSION_NAME}:0.1

# Attach to session
echo -e "${GREEN}Dashboard started!${NC}"
sleep 1
tmux attach-session -t ${SESSION_NAME}