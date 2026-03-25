#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: ./restore.sh <backup_file.sql>"
  exit 1
fi
docker exec -i lifecycleofacreator-db-1 psql -U postgres lifecycle < "$1"
