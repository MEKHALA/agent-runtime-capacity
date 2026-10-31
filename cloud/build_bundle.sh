#!/usr/bin/env bash
# Build a deployment zip for Lambda and AgentCore Runtime direct code deployment,
# without Docker: download pre-built Linux wheels for the target architecture.
#
#   cloud/build_bundle.sh arm64     # AgentCore (arm64 only) and Lambda arm64
#   cloud/build_bundle.sh x86_64    # Lambda x86_64
#
# Output: cloud/build/bench-<arch>.zip  (libraries and code at the zip root)
set -euo pipefail
ARCH="${1:-arm64}"
case "$ARCH" in
  arm64)  PLAT=manylinux2014_aarch64 ;;
  x86_64) PLAT=manylinux2014_x86_64 ;;
  *) echo "arch must be arm64 or x86_64" >&2; exit 2 ;;
esac
PYVER=3.12
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/cloud/build/$ARCH"
rm -rf "$OUT" && mkdir -p "$OUT/pkg"

"$PY" -m pip install --quiet --target "$OUT/pkg" \
  --platform "$PLAT" --platform manylinux_2_28_${PLAT#manylinux2014_} \
  --implementation cp --python-version "$PYVER" \
  --only-binary=:all: -r "$ROOT/cloud/requirements.txt"

cp -r "$ROOT/agentprof" "$OUT/pkg/"
mkdir -p "$OUT/pkg/scripts"
cp "$ROOT"/scripts/{measure_framework_cpu,measure_strands_cpu,bedrock_transport_mock}.py "$OUT/pkg/scripts/"
cp "$ROOT"/cloud/{bench,lambda_function,main}.py "$OUT/pkg/"
find "$OUT/pkg" -name '__pycache__' -type d -prune -exec rm -rf {} +

# Native-binary architecture check: AgentCore rejects any non-arm64 .so.
BAD=$(find "$OUT/pkg" -name '*.so' -exec file {} + | grep -v -E "$( [ "$ARCH" = arm64 ] && echo 'aarch64|ARM aarch64' || echo 'x86-64')" || true)
if [ -n "$BAD" ]; then echo "wrong-architecture binaries:"; echo "$BAD"; exit 1; fi

(cd "$OUT/pkg" && "$PY" - "$ROOT/cloud/build/bench-$ARCH.zip" <<'EOF'
import os, sys, zipfile
dest = sys.argv[1]
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk("."):
        for f in files:
            p = os.path.join(base, f)
            z.write(p, os.path.relpath(p, "."))
EOF
)
du -sh "$OUT/pkg" "$ROOT/cloud/build/bench-$ARCH.zip"
