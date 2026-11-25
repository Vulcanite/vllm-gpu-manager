import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'intro',
    'install',
    'quickstart',
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/downloads',
        'guides/deploy',
        'guides/gpu-and-vram',
        'guides/proxy',
        'guides/service',
        'guides/monitoring',
        'guides/command-import',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: [
        'reference/cli',
        'reference/api',
        'reference/settings',
        'reference/hardware-compatibility',
      ],
    },
    'security',
    'architecture',
    'contributing',
    'changelog',
  ],
};

export default sidebars;
