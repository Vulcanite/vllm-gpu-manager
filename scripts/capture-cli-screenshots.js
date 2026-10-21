const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function renderTerminal(page, title, command, output, outputPath) {
  const html = `
  <!DOCTYPE html>
  <html>
  <head>
    <style>
      body {
        margin: 0;
        padding: 24px;
        background: transparent;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: ui-monospace, 'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'IBM Plex Mono', monospace;
      }
      .window {
        width: 1000px;
        background: #0B0F17;
        border: 1px solid #1E293B;
        border-radius: 12px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.6), 0 0 0 1px rgba(6,182,212,0.1);
        overflow: hidden;
      }
      .titlebar {
        background: #070A0F;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid #1E293B;
      }
      .buttons {
        display: flex;
        gap: 8px;
        margin-right: 16px;
      }
      .btn {
        width: 12px;
        height: 12px;
        border-radius: 50%;
      }
      .btn.red { background: #EF4444; }
      .btn.yellow { background: #F59E0B; }
      .btn.green { background: #10B981; }
      .title {
        color: #94A3B8;
        font-size: 13px;
        font-weight: 500;
      }
      .body {
        padding: 20px 24px;
        color: #F1F5F9;
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
      }
      .prompt {
        color: #06B6D4;
        font-weight: 700;
      }
      .cmd {
        color: #F8FAFC;
        font-weight: 600;
      }
      .cyan { color: #06B6D4; font-weight: 600; }
      .green { color: #10B981; font-weight: 600; }
      .yellow { color: #F59E0B; }
      .purple { color: #A855F7; }
      .muted { color: #64748B; }
    </style>
  </head>
  <body>
    <div class="window">
      <div class="titlebar">
        <div class="buttons">
          <div class="btn red"></div>
          <div class="btn yellow"></div>
          <div class="btn green"></div>
        </div>
        <div class="title">${title}</div>
      </div>
      <div class="body"><span class="prompt">admin@h100-node-01:~$</span> <span class="cmd">${command}</span>\n\n${output}</div>
    </div>
  </body>
  </html>
  `;

  await page.setContent(html, { waitUntil: 'domcontentloaded' });
  await new Promise(r => setTimeout(r, 100));
  const element = await page.$('.window');
  await element.screenshot({ path: outputPath, omitBackground: true });
}

async function run() {
  const outputDir = path.join(__dirname, '..', 'static', 'img', 'screenshots', 'cli');
  fs.mkdirSync(outputDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    defaultViewport: { width: 1100, height: 800, deviceScaleFactor: 2 }
  });

  const page = await browser.newPage();

  // 1. GPU List
  console.log('Rendering 01-cli-gpu-list.png...');
  const gpuListOutput = `<span class="muted">+----+-----------------------------+------------+-----------+----------+--------+</span>
| <span class="cyan">ID</span> | <span class="cyan">NAME</span>                        | <span class="cyan">TOTAL VRAM</span> | <span class="cyan">USED VRAM</span> | <span class="cyan">FREE</span>     | <span class="cyan">UTIL %</span> |
<span class="muted">+----+-----------------------------+------------+-----------+----------+--------+</span>
| 0  | NVIDIA H100 80GB HBM3       | 80.00 GiB  | 34.18 GiB | 45.82 GiB| <span class="green">68%</span>    |
| 1  | NVIDIA H100 80GB HBM3       | 80.00 GiB  | 34.18 GiB | 45.82 GiB| <span class="green">68%</span>    |
<span class="muted">+----+-----------------------------+------------+-----------+----------+--------+</span>

<span class="purple">Active Compute Processes:</span>
  • <span class="cyan">PID 24819</span> (user: <span class="green">vllm-manager</span>): python3 (vLLM::Engine) on GPU 0 [34.18 GiB]
  • <span class="cyan">PID 24820</span> (user: <span class="green">vllm-manager</span>): python3 (vLLM::Worker) on GPU 1 [34.18 GiB]`;
  await renderTerminal(page, 'vllm-manager gpu list', 'vllm-manager gpu list', gpuListOutput, path.join(outputDir, '01-cli-gpu-list.png'));

  // 2. Doctor Diagnostics
  console.log('Rendering 02-cli-doctor.png...');
  const doctorOutput = `<span class="green">✔</span> Operating system         : linux (amd64)
<span class="green">✔</span> nvidia-smi               : /usr/bin/nvidia-smi (Driver 550.90.07, CUDA 12.4)
<span class="green">✔</span> NVIDIA GPUs              : 2 GPU(s) detected (Compute Capability 9.0 Hopper)
<span class="green">✔</span> Docker CLI               : /usr/bin/docker (Version 27.1.1)
<span class="green">✔</span> Docker API socket        : unix:///var/run/docker.sock (ping OK)
<span class="green">✔</span> NVIDIA Container Toolkit : GPU container runtime verified
<span class="green">✔</span> Hugging Face Cache       : /var/lib/vllm-manager/.cache/huggingface (1.52 TiB free)
<span class="green">✔</span> Service environment      : /etc/vllm-manager/service.env (0640 root:vllm-manager)

<span class="green">All preflight diagnostics PASSED. Host is ready for serving.</span>`;
  await renderTerminal(page, 'vllm-manager doctor', 'vllm-manager doctor', doctorOutput, path.join(outputDir, '02-cli-doctor.png'));

  // 3. Model Start
  console.log('Rendering 03-cli-model-start.png...');
  const modelStartOutput = `<span class="cyan">[VRAM Preflight]</span> Validating Qwen/Qwen2.5-Coder-32B-Instruct:
  • Architecture         : Qwen2ForCausalLM (64 layers, 8 KV heads, 128 head_dim)
  • Tensor Parallel Size : 2 GPUs (GPU 0, GPU 1)
  • Model Weights Size   : 63.18 GiB total (~34.12 GiB / GPU with 1.08x safety)
  • KV Cache per Token   : 32.00 KiB / token / GPU (fp8 precision)
  • Max Context Budget   : 32768 tokens (1.00 GiB KV cache / GPU)
  • Target Free VRAM     : 45.82 GiB available > 35.12 GiB required -> <span class="green">FIT VERIFIED</span>

<span class="green">✔</span> Created container <span class="cyan">vllm-Qwen2.5-Coder-32B-Instruct-8080</span> (id: c1f90a2b8e34)
<span class="green">✔</span> Published port <span class="yellow">127.0.0.1:8080</span> -> vLLM OpenAI Engine
<span class="green">✔</span> Updated proxy routing table: <span class="purple">"Qwen/Qwen2.5-Coder-32B-Instruct"</span> -> <span class="yellow">:8080</span>
<span class="green">✔</span> Server started successfully. Streaming health check: <span class="green">HEALTHY (ready)</span>`;
  await renderTerminal(page, 'vllm-manager model start', 'vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct --port 8080 --gpus 2 --kv-cache-dtype fp8', modelStartOutput, path.join(outputDir, '03-cli-model-start.png'));

  await browser.close();
  console.log('CLI terminal screenshots generated successfully!');
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
