#!/bin/bash
set -e

echo "Building React Frontend..."
cd frontend
npm install
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
echo "   (The dashboard will be available at https://localhost:9090)"
echo ""
echo "To run it in the background:"
echo "   ./manager.sh start 9090"
