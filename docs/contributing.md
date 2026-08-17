---
title: Contributing
sidebar_position: 6
---

# Contributing

Thank you for your interest in contributing to **vLLM GPU Manager**!

---

## Development Workflow

### 1. Prerequisites
- **Go**: 1.23+
- **Node.js**: 20+
- **Docker**: Docker Engine with NVIDIA Container Toolkit

### 2. Fork and Clone
```bash
git clone https://github.com/Vulcanite/vllm-gpu-manager.git
cd vllm-gpu-manager
```

### 3. Frontend Development
Run the Vite development server with proxying to the backend:

```bash
cd frontend
npm install
npm run dev
```

### 4. Running Backend Tests
Execute unit tests with the Go race detector enabled:

```bash
go test -v -race ./...
```

### 5. Building Documentation
Typecheck and build the Docusaurus documentation site locally:

```bash
npm install
npm run typecheck
npm run docs:build
```

---

## Pull Request Guidelines

1. **Keep Pull Requests Focused**: Target a specific feature, bug fix, or documentation improvement.
2. **Add Tests**: Include unit tests in `api/` or root `*_test.go` files covering new logic or bug reproductions.
3. **Preserve Compatibility**: Ensure CLI flag syntax and REST API schemas remain backwards compatible.
4. **License Compliance**: Dependencies must be compatible with the **Apache-2.0** license. Run `./scripts/check-licenses.sh` to verify.
