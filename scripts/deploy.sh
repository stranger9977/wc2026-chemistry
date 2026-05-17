#!/usr/bin/env bash
set -euo pipefail
python3 -m scripts.build --heuristic-vaep --no-fetch
python3 -m scripts.export
git add outputs/chemistry.json exports/
git commit -m "Refresh: chemistry build + exports" || echo "nothing to commit"
git push
