#!/bin/bash
set -e

echo "Building React Frontend..."
cd frontend
npm run build
cd ..

echo "Building Golang Backend Binary..."
CGO_ENABLED=0 go build -o vllm-manager main.go

echo ""
echo "✅ Build complete! You can now run the standalone server:"
echo "   ./vllm-manager"
echo ""
echo "To run it in the background:"
echo "   nohup ./vllm-manager > vllm-manager.log 2>&1 &"
