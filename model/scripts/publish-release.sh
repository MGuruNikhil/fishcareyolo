#!/bin/bash
#
# Publish model binaries to GitHub releases
#
# Usage:
#   ./scripts/publish-release.sh dev [version]   # Create dev release (e.g., dev-v1.0.0)
#   ./scripts/publish-release.sh prod [version]  # Create prod release (e.g., v1.0.0)
#
# Examples:
#   ./scripts/publish-release.sh dev 1.0.0
#   ./scripts/publish-release.sh prod 1.0.0
#

set -e

CHANNEL="${1:-}"
VERSION="${2:-}"

if [[ -z "$CHANNEL" ]]; then
    echo "Error: Channel required (dev or prod)"
    echo "Usage: $0 <dev|prod> <version>"
    exit 1
fi

if [[ "$CHANNEL" != "dev" && "$CHANNEL" != "prod" ]]; then
    echo "Error: Channel must be 'dev' or 'prod'"
    exit 1
fi

if [[ -z "$VERSION" ]]; then
    echo "Error: Version required (e.g., 1.0.0)"
    echo "Usage: $0 <dev|prod> <version>"
    exit 1
fi

# Set tag based on channel
if [[ "$CHANNEL" == "dev" ]]; then
    TAG="dev-v${VERSION}"
    PRERELEASE="--prerelease"
    TITLE="Dev Release v${VERSION}"
else
    TAG="v${VERSION}"
    PRERELEASE=""
    TITLE="Release v${VERSION}"
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="${MODEL_DIR}/runs/detect/fish_disease/weights"
TFLITE_DIR="${WEIGHTS_DIR}/best_saved_model"

# Check if weights exist
if [[ ! -d "$WEIGHTS_DIR" ]]; then
    echo "Error: Weights directory not found: $WEIGHTS_DIR"
    echo "Run training first: uv run mina-train"
    exit 1
fi

# Files to upload
FILES=(
    "${TFLITE_DIR}/best_full_integer_quant.tflite"
    "${TFLITE_DIR}/best_float16.tflite"
    "${WEIGHTS_DIR}/best.onnx"
    "${WEIGHTS_DIR}/best.pt"
    "${TFLITE_DIR}/metadata.yaml"
)

# Check all files exist
echo "Checking files..."
for file in "${FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: File not found: $file"
        exit 1
    fi
    echo "  ✓ $(basename "$file")"
done

# Check if release already exists
if gh release view "$TAG" &>/dev/null; then
    echo ""
    echo "Release $TAG already exists."
    read -p "Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing release..."
        gh release delete "$TAG" --yes
        git push --delete origin "$TAG" 2>/dev/null || true
    else
        echo "Aborted."
        exit 1
    fi
fi

# Create release notes
NOTES=$(cat <<EOF
## Fish Disease Detection Model

**Channel:** ${CHANNEL}
**Version:** ${VERSION}

### Files

| File | Description | Size |
|------|-------------|------|
| \`best_full_integer_quant.tflite\` | Int8 quantized TFLite (recommended for mobile) | $(du -h "${TFLITE_DIR}/best_full_integer_quant.tflite" | cut -f1) |
| \`best_float16.tflite\` | Float16 TFLite | $(du -h "${TFLITE_DIR}/best_float16.tflite" | cut -f1) |
| \`best.onnx\` | ONNX model | $(du -h "${WEIGHTS_DIR}/best.onnx" | cut -f1) |
| \`best.pt\` | PyTorch weights | $(du -h "${WEIGHTS_DIR}/best.pt" | cut -f1) |
| \`metadata.yaml\` | Model metadata | $(du -h "${TFLITE_DIR}/metadata.yaml" | cut -f1) |

### Model Specs

- **Input:** 640×640×3 RGB image
- **Classes:** bacterial_infection, fungal_infection, healthy, parasite, white_tail
- **NMS:** Not included (implement in app)

### Usage in Expo App

Download the TFLite model:
\`\`\`bash
curl -L -o model.tflite https://github.com/fishcareyolo/fishcareyolo/releases/download/${TAG}/best_full_integer_quant.tflite
\`\`\`
EOF
)

echo ""
echo "Creating release: $TAG"
echo "Title: $TITLE"
echo ""

# Create the release
gh release create "$TAG" \
    --title "$TITLE" \
    --notes "$NOTES" \
    $PRERELEASE \
    "${FILES[@]}"

echo ""
echo "✓ Release created: $TAG"
echo ""
echo "View at: https://github.com/fishcareyolo/fishcareyolo/releases/tag/${TAG}"
