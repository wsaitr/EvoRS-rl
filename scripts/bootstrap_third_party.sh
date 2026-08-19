#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/third_party"

if [ ! -d CoMLRL ]; then
  git clone https://github.com/OpenMLRL/CoMLRL.git
fi
if [ ! -d MARTI ]; then
  git clone https://github.com/TsinghuaC3I/MARTI.git
fi

echo "Formal experiments must pin exact commits."
echo "CoMLRL: cd third_party/CoMLRL && git rev-parse HEAD"
echo "MARTI:  cd third_party/MARTI && git rev-parse HEAD"
