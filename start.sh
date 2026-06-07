#!/bin/bash
docker start nginx qdrant_index deploy_qwenie tensorrt_bge-m3
# docker start qdrant_index deploy_saulie tensorrt_bge-m3

# 1. Use sh -c to allow redirection
# 2. Use python -u for unbuffered output (crucial for seeing logs immediately)
# 3. Redirect stdout (1) and stderr (2) to /proc/1/fd/1
# >> these allow serve.py's logs to be redirected to show up in the container's logs
docker exec -d tensorrt_bge-m3 sh -c "python -u serve.py >> /proc/1/fd/1 2>&1"

echo "All services are up.."
