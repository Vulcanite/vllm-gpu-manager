#!/bin/bash
set -e

echo "Building React Frontend..."
cd frontend
npm run build
cd ..

echo "Building Golang Backend Binary..."
CGO_ENABLED=0 go build -o vllm-manager .

echo ""
echo "✅ Build complete! You can now run the CLI:"
echo "   ./vllm-manager"
echo ""
echo "Or start the dashboard:"
echo "   ./vllm-manager --dashboard --port 9090"
echo ""
echo "To run it in the background:"
echo "   nohup ./vllm-manager --dashboard --port 9090 > vllm-manager.log 2>&1 &"
