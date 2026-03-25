#!/bin/bash
docker exec lifecycleofacreator-db-1 pg_dump -U postgres lifecycle > backup_$(date +%Y%m%d_%H%M%S).sql
