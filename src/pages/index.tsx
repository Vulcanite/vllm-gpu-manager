import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const asciiLogo = String.raw` _   _ _     _     __  __    ____ ____  _   _ 
| | | | |   | |   |  \/  |  / ___|  _ \| | | |
| | | | |   | |   | |\/| | | |  _| |_) | | | |
| |_| | |___| |___| |  | | | |_| |  __/| |_| |
 \___/|_____|_____|_|  |_|  \____|_|    \___/ `;

const features = [
  [
    '01',
    'Preflight Guardrails',
    'Calculate exact weight and KV cache memory requirements before starting containers to prevent runtime OOM failures.',
  ],
  [
    '02',
    'OpenAI Proxy Gateway',
    'Route dynamic /v1/ requests to underlying active model containers with Bearer token authentication and per-user tracking.',
  ],
  [
    '03',
    'Multi-GPU Placement',
    'Fine-tune tensor and pipeline parallelism, explicit GPU device pinning, and topology link discovery from one interface.',
  ],
  [
    '04',
    'Single Binary & Packages',
    'Zero runtime dependencies. Embedded React console, full CLI, systemd integration, and native Debian & RPM packages.',
  ],
  [
    '05',
    'Hugging Face Cache Ops',
    'Track background download progress, inspect local cache sizes, and safely delete models via sandboxed helper containers.',
  ],
  [
    '06',
    'Prometheus & DCGM Sync',
    'Automatically mirror active model endpoints into Prometheus file-SD targets with preconfigured Grafana dashboards.',
  ],
];

function Homepage(): ReactNode {
  return (
    <Layout
      title="vLLM GPU Manager — Operations Console & OpenAI Proxy"
      description="Lightweight single-binary CLI and web console for managing vLLM OpenAI-compatible servers on NVIDIA GPUs."
    >
      <main>
        <section className={styles.hero}>
          <div className={styles.coordinate}>LINUX / NVIDIA / DOCKER / VLLM</div>
          <div className={styles.heroGrid}>
            <div className={styles.heroCopy}>
              <span className={styles.eyebrow}>GPU Operations Made Predictable</span>
              <Heading as="h1">
                Stop guessing VRAM.<br />Start serving models.
              </Heading>
              <p>
                vLLM GPU Manager runs OpenAI-compatible LLM servers as isolated Docker containers on NVIDIA GPU hosts—with
                rigorous launch-time VRAM preflight, multi-user proxy routing, and real-time telemetry.
              </p>
              <div className={styles.actions}>
                <Link className={clsx('button', styles.primaryAction)} to="/docs/getting-started/quickstart">
                  Quickstart in 5 mins <span aria-hidden="true">→</span>
                </Link>
                <Link className={clsx('button', styles.secondaryAction)} to="/docs/reference/cli">
                  CLI Reference
                </Link>
              </div>
            </div>
            <div className={styles.terminal} aria-label="vLLM GPU Manager terminal preview">
              <div className={styles.terminalBar}>
                <span>vllm-manager — h100-node-01</span>
                <span className={styles.status}>● ONLINE</span>
              </div>
              <pre className={styles.ascii} aria-hidden="true">
                {asciiLogo}
              </pre>
              <div className={styles.commandLine}>
                <span>$</span> vllm-manager model start Qwen/Qwen2.5-Coder-32B-Instruct --port 8080 --gpus 2
              </div>
              <div className={styles.routeLine}>
                <span>GPU 0,1 (80GB)</span>
                <i />
                <span>VRAM Preflight: 34.2 GiB / 72.0 GiB (47.5%)</span>
                <i />
                <strong>PASS</strong>
              </div>
            </div>
          </div>
          <div className={styles.heroIndex}>[ NVIDIA H100 / A100 / L40S / RTX 4090 / 6000 Ada ]</div>
        </section>

        <section className={styles.featureSection} aria-labelledby="features-heading">
          <div className={styles.sectionHeading}>
            <span>PRODUCTION CAPABILITIES</span>
            <Heading as="h2" id="features-heading">
              Engineered for Real-World GPU Serving
            </Heading>
          </div>
          <div className={styles.featureGrid}>
            {features.map(([index, title, body]) => (
              <article className={styles.feature} key={index}>
                <span className={styles.featureIndex}>{index}</span>
                <Heading as="h3">{title}</Heading>
                <p>{body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.commandSection} aria-labelledby="commands-heading">
          <div className={styles.commandSectionGrid}>
            <div>
              <span className={styles.eyebrow}>SSH & Automation First</span>
              <Heading as="h2" id="commands-heading">
                Full CLI Control.<br />Zero GUI Dependency.
              </Heading>
              <p style={{ color: 'var(--site-muted)', marginTop: '1rem', lineHeight: '1.6' }}>
                Every operation available in the web console can be executed directly over SSH without running the dashboard daemon.
              </p>
            </div>
            <div className={styles.commandStack}>
              <code>
                <b>STATUS</b> vllm-manager gpu list
              </code>
              <code>
                <b>DEPLOY</b> vllm-manager model start deepseek-ai/DeepSeek-V4 --port 8080 --gpus 4
              </code>
              <code>
                <b>ROUTING</b> curl -H "Authorization: Bearer sk-..." http://localhost:9090/v1/chat/completions
              </code>
              <code>
                <b>DOCTOR</b> vllm-manager doctor
              </code>
            </div>
          </div>
        </section>

        <section className={styles.finalCta}>
          <span>READY TO DEPLOY ON YOUR GPU SERVERS?</span>
          <Link to="/docs/getting-started/installation">
            Install vLLM GPU Manager <span aria-hidden="true">↗</span>
          </Link>
        </section>
      </main>
    </Layout>
  );
}

export default Homepage;
