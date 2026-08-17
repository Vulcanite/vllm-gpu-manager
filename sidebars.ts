import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'overview',
    {
      type: 'category',
      label: 'Getting started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/core-concepts',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/deploying-models',
        'guides/gpu-placement',
        'guides/vram-calculation',
        'guides/huggingface-cache',
        'guides/traffic-routing',
        'guides/monitoring',
        'guides/systemd-service',
        'guides/command-import',
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
    'architecture',
    'security',
    'contributing',
    'changelog',
  ],
};

export default sidebars;
