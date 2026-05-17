#!/usr/bin/env bash
set -euo pipefail
rm -rf _site
mkdir -p _site
cp site/*.html site/*.css site/*.js _site/
cp -r outputs exports assets _site/ 2>/dev/null || true
echo "Serving _site/ at http://localhost:8080/"
python3 -m http.server -d _site 8080
