import React, {useState, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './index.module.css';

const pillars = [
  {
    icon: '⚡',
    title: 'VRAM Preflight Guardrails',
    desc: 'Calculates exact parameter weights and token KV cache requirements before container launch to eliminate CUDA OOM panics.',
  },
  {
    icon: '🔀',
    title: 'OpenAI Reverse Proxy',
    desc: 'Centralized /v1/ gateway that validates Bearer user tokens, inspects target models, rewrites headers, and streams SSE responses.',
  },
  {
    icon: '🧩',
    title: 'Multi-GPU Placement',
    desc: 'Fine-grained tensor parallelism, pipeline parallelism, and explicit device pinning with NVLink topology discovery.',
  },
  {
    icon: '📦',
    title: 'Hub Cache & Packaging',
    desc: 'Track model download containers, inspect disk footprints, and install via native Linux Debian & RPM packages.',
  },
];

function Homepage(): ReactNode {
  const [activeTab, setActiveTab] = useState<'web' | 'cli' | 'vram' | 'proxy'>('web');

  const webImg = useBaseUrl('/img/screenshots/web/01-dashboard-overview.png');
  const cliImg = useBaseUrl('/img/screenshots/cli/01-cli-gpu-list.png');
  const vramImg = useBaseUrl('/img/diagrams/vram-preflight-engine.png');
  const proxyImg = useBaseUrl('/img/diagrams/proxy-routing-flow.png');
  const archImg = useBaseUrl('/img/diagrams/system-architecture.png');

  return (
    <Layout
      title="vLLM GPU Manager — Operations Console & OpenAI Proxy for NVIDIA GPUs"
      description="Standalone CLI and real-time web management console for vLLM OpenAI-compatible model serving on NVIDIA GPUs."
    >
      <main>
        <section className={styles.hero}>
          <div className={styles.heroContent}>
            <div className={styles.badge}>
              <span className={styles.badgeDot} />
              Dual Interface · Web Console &amp; SSH CLI
            </div>
            <Heading as="h1" className={styles.heroTitle}>
              Production vLLM Operations &amp; OpenAI Proxy <span className={styles.gradientText}>for NVIDIA GPUs</span>
            </Heading>
            <p className={styles.heroSubtitle}>
              Run local LLM servers inside isolated Docker containers with automated VRAM preflight math,
              multi-GPU tensor parallelism, and a dynamic OpenAI reverse proxy.
            </p>
            <div className={styles.ctaGroup}>
              <Link className={clsx('button', styles.btnPrimary)} to="/docs/getting-started/quickstart">
                Get Started (5 Min) →
              </Link>
              <Link className={clsx('button', styles.btnSecondary)} to="/docs/web-guide/overview">
                Explore Web Console Guide
              </Link>
              <Link className={clsx('button', styles.btnSecondary)} to="/docs/cli-guide/overview">
                Explore CLI Guide
              </Link>
            </div>

            {/* Interactive Showcase Preview */}
            <div className={styles.showcaseWrapper}>
              <div className={styles.showcaseNav}>
                <div className={styles.tabButtons}>
                  <button
                    className={clsx(styles.tabBtn, activeTab === 'web' && styles.tabBtnActive)}
                    onClick={() => setActiveTab('web')}
                  >
                    🖥️ Web Console
                  </button>
                  <button
                    className={clsx(styles.tabBtn, activeTab === 'cli' && styles.tabBtnActive)}
                    onClick={() => setActiveTab('cli')}
                  >
                    ⚡ SSH CLI
                  </button>
                  <button
                    className={clsx(styles.tabBtn, activeTab === 'vram' && styles.tabBtnActive)}
                    onClick={() => setActiveTab('vram')}
                  >
                    📊 VRAM Engine
                  </button>
                  <button
                    className={clsx(styles.tabBtn, activeTab === 'proxy' && styles.tabBtnActive)}
                    onClick={() => setActiveTab('proxy')}
                  >
                    🔀 Proxy Flow
                  </button>
                </div>
                <span style={{ fontSize: '0.78rem', color: 'var(--site-muted)', fontFamily: 'var(--site-mono)' }}>
                  NVIDIA H100 / A100 / L40S / RTX 4090
                </span>
              </div>
              <div style={{ padding: '0.75rem', background: '#05080C' }}>
                {activeTab === 'web' && (
                  <img src={webImg} alt="vLLM GPU Manager Web Console" className={styles.showcaseImage} />
                )}
                {activeTab === 'cli' && (
                  <img src={cliImg} alt="vLLM GPU Manager CLI Interface" className={styles.showcaseImage} />
                )}
                {activeTab === 'vram' && (
                  <img src={vramImg} alt="VRAM Preflight Calculation Engine" className={styles.showcaseImage} />
                )}
                {activeTab === 'proxy' && (
                  <img src={proxyImg} alt="OpenAI Reverse Proxy Architecture" className={styles.showcaseImage} />
                )}
              </div>
            </div>
          </div>
        </section>

        {/* 4 Core Pillars */}
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionEyebrow}>Core Capabilities</span>
            <Heading as="h2" className={styles.sectionTitle}>
              Engineered for Enterprise GPU Servers
            </Heading>
          </div>
          <div className={styles.grid4}>
            {pillars.map((p, idx) => (
              <div key={idx} className={styles.card}>
                <div className={styles.cardIcon}>{p.icon}</div>
                <Heading as="h3">{p.title}</Heading>
                <p>{p.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Architecture Section */}
        <section className={styles.archSection}>
          <div className={styles.archGrid}>
            <div>
              <span className={styles.sectionEyebrow}>End-to-End Design</span>
              <Heading as="h2" style={{ fontSize: '2.4rem', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '1.25rem' }}>
                Single-Binary Simplicity.<br />Zero Extra Runtimes.
              </Heading>
              <p style={{ color: 'var(--site-muted)', lineHeight: '1.7', fontSize: '1.05rem', marginBottom: '1.75rem' }}>
                Built in pure Go with an embedded React web application. Directly talks to the Docker Engine socket and queries
                NVIDIA telemetry to orchestrate standard <code>vllm/vllm-openai</code> containers without complex Kubernetes dependencies.
              </p>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <Link className={clsx('button', styles.btnPrimary)} to="/docs/architecture">
                  View Architecture Deep Dive →
                </Link>
                <Link className={clsx('button', styles.btnSecondary)} to="/docs/prerequisites">
                  Prerequisites
                </Link>
              </div>
            </div>
            <div>
              <img src={archImg} alt="vLLM GPU Manager Architecture Diagram" className={styles.archImage} />
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <section className={styles.finalSection}>
          <Heading as="h2">Ready to Deploy on Your GPU Infrastructure?</Heading>
          <p style={{ color: 'var(--site-muted)', fontSize: '1.15rem', marginBottom: '2rem' }}>
            Install native Debian/RPM packages or run the standalone binary in seconds.
          </p>
          <Link className={clsx('button', styles.btnPrimary)} to="/docs/getting-started/installation">
            Install vLLM GPU Manager →
          </Link>
        </section>
      </main>
    </Layout>
  );
}

export default Homepage;
