import React, {useState, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './index.module.css';

const BLOG_URL = 'https://amishtp.tech/blog/vllm-gpu-manager';

const features = [
  {
    title: 'Checks VRAM before launch',
    desc: 'Estimates model weights plus KV cache for your context length and refuses a launch that will not fit in the GPU memory you gave it.',
  },
  {
    title: 'One OpenAI-compatible endpoint',
    desc: 'A /v1/ proxy routes requests to the right model container, with per-user API keys and request counts.',
  },
  {
    title: 'Pick the GPUs',
    desc: 'Tensor or pipeline parallelism, pinning to exact GPU IDs, and a view of how the GPUs are connected.',
  },
  {
    title: 'Manages the model cache',
    desc: 'Downloads Hugging Face models in the background, shows progress and failures, and deletes cached models safely.',
  },
];

type Tab = 'web' | 'cli' | 'vram' | 'proxy';

const tabs: {id: Tab; label: string}[] = [
  {id: 'web', label: 'Dashboard'},
  {id: 'cli', label: 'CLI'},
  {id: 'vram', label: 'VRAM check'},
  {id: 'proxy', label: 'Proxy'},
];

function Homepage(): ReactNode {
  const [activeTab, setActiveTab] = useState<Tab>('web');

  const images: Record<Tab, {src: string; alt: string; note?: string}> = {
    web: {
      src: useBaseUrl('/img/screenshots/web/01-dashboard-overview.png'),
      alt: 'vLLM GPU Manager dashboard with one model running on two GPUs',
      note: 'Demo build with sample data.',
    },
    cli: {
      src: useBaseUrl('/img/screenshots/cli/01-cli-gpu-list.png'),
      alt: 'vllm-manager gpu list output',
      note: 'Illustrative mock-up from sample data; real output differs in layout.',
    },
    vram: {src: useBaseUrl('/img/diagrams/vram-preflight-engine.png'), alt: 'How the VRAM check works'},
    proxy: {src: useBaseUrl('/img/diagrams/proxy-routing-flow.png'), alt: 'How the /v1/ proxy routes requests'},
  };
  const active = images[activeTab];

  return (
    <Layout
      title="vLLM GPU Manager: run vLLM on your own GPU server"
      description="One binary with a CLI and web dashboard to run vLLM in Docker on NVIDIA GPUs, with VRAM checks before launch and an OpenAI-compatible proxy."
    >
      <main>
        <section className={styles.hero}>
          <div className={styles.heroContent}>
            <Heading as="h1" className={styles.heroTitle}>
              Run vLLM on your own GPU server
            </Heading>
            <p className={styles.heroSubtitle}>
              One binary with a CLI and a web dashboard. It starts vLLM in Docker, checks that a model fits in GPU
              memory before launching it, and puts every model behind one OpenAI-compatible endpoint.
            </p>
            <div className={styles.ctaGroup}>
              <Link className={clsx('button', styles.btnPrimary)} to="/docs/quickstart">
                Get started
              </Link>
              <span className={styles.ctaLinks}>
                <Link to="/docs/install">Install</Link>
                <span aria-hidden="true">·</span>
                <a href={BLOG_URL}>Why I built it</a>
              </span>
            </div>

            <div className={styles.showcaseWrapper}>
              <div className={styles.showcaseNav} role="tablist">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    role="tab"
                    aria-selected={activeTab === tab.id}
                    className={clsx(styles.tabBtn, activeTab === tab.id && styles.tabBtnActive)}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              <div className={styles.showcaseBody}>
                <a href={active.src} target="_blank" rel="noopener noreferrer" title="Open full size">
                  <img src={active.src} alt={active.alt} className={styles.showcaseImage} />
                </a>
                {active.note && <p className={styles.showcaseNote}>{active.note}</p>}
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <Heading as="h2" className={styles.sectionTitle}>
            What it does
          </Heading>
          <div className={styles.grid4}>
            {features.map((f) => (
              <div key={f.title} className={styles.card}>
                <Heading as="h3">{f.title}</Heading>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
          <p className={styles.scopeNote}>
            It is deliberately small: one Linux host, NVIDIA GPUs and Docker. It is not a cluster scheduler.
          </p>
        </section>
      </main>
    </Layout>
  );
}

export default Homepage;
