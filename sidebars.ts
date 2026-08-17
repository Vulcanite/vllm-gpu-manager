import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'overview',
    'prerequisites',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/core-concepts',
      ],
    },
    {
      type: 'category',
      label: 'Web Console Guide',
      collapsed: false,
      items: [
        'web-guide/overview',
        'web-guide/telemetry',
        'web-guide/deploying-models',
        'web-guide/model-management',
        'web-guide/cache-and-downloads',
        'web-guide/proxy-gateway',
        'web-guide/settings-and-auth',
      ],
    },
    {
      type: 'category',
      label: 'CLI Operations Guide',
      collapsed: false,
      items: [
        'cli-guide/overview',
        'cli-guide/gpu-management',
        'cli-guide/model-operations',
        'cli-guide/routing-and-proxy',
        'cli-guide/automation-and-json',
        'cli-guide/service-management',
      ],
    },
    {
      type: 'category',
      label: 'Deep Dives & Ops',
      items: [
        'guides/vram-calculation',
        'guides/gpu-placement',
        'guides/traffic-routing',
        'guides/monitoring',
        'guides/command-import',
      ],
    },
    {
      type: 'category',
      label: 'Architecture & Security',
      items: [
        'architecture',
        'security',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli',
        'reference/api',
        'reference/settings',
        'reference/hardware-compatibility',
      ],
    },
    'contributing',
    'changelog',
  ],
};

export default sidebars;
