import type {Config} from '@docusaurus/types';
import type {Options, ThemeConfig} from '@docusaurus/preset-classic';
import {themes as prismThemes} from 'prism-react-renderer';
import packageJson from './package.json';

const organizationName = process.env.GITHUB_REPOSITORY_OWNER ?? 'Vulcanite';
const projectName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? 'vllm-gpu-manager';
const repositoryUrl = `https://github.com/${organizationName}/${projectName}`;
const editBranch = process.env.DOCS_EDIT_BRANCH ?? 'main';
const isUserSite = projectName === `${organizationName}.github.io`;

const config: Config = {
  title: 'vLLM GPU Manager',
  tagline: 'Lightweight operations console, VRAM preflight engine, and OpenAI proxy for vLLM on NVIDIA GPUs.',
  favicon: 'img/favicon.svg',
  url: process.env.DOCS_URL ?? `https://${organizationName}.github.io`,
  baseUrl: process.env.DOCS_BASE_URL ?? (isUserSite ? '/' : `/${projectName}/`),
  organizationName,
  projectName,
  trailingSlash: false,
  onBrokenLinks: 'throw',
  markdown: {
    mermaid: true,
    hooks: {onBrokenMarkdownLinks: 'warn'},
  },
  themes: ['@docusaurus/theme-mermaid'],
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          showLastUpdateTime: true,
          editUrl: `${repositoryUrl}/edit/${editBranch}/`,
        },
        blog: false,
        theme: {customCss: './src/css/custom.css'},
        sitemap: {changefreq: 'weekly', priority: 0.5},
      } satisfies Options,
    ],
  ],
  themeConfig: {
    image: 'img/social-card.svg',
    metadata: [
      {name: 'theme-color', content: '#10b981'},
      {
        name: 'description',
        content: 'Documentation for vLLM GPU Manager — single-binary CLI and web console for vLLM on NVIDIA GPUs with VRAM preflight and OpenAI proxy.',
      },
    ],
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'vLLM GPU Manager',
      logo: {alt: 'vLLM GPU Manager Logo', src: 'img/logo.svg'},
      items: [
        {to: '/docs/getting-started/quickstart', label: 'Get Started', position: 'left'},
        {to: '/docs/guides/deploying-models', label: 'Guides', position: 'left'},
        {to: '/docs/reference/cli', label: 'Reference', position: 'left'},
        {to: '/docs/architecture', label: 'Architecture', position: 'left'},
        {to: '/docs/changelog', label: 'Changelog', position: 'left'},
        {type: 'html', value: `<span style="font-family: var(--site-mono); font-size: 0.8rem; padding: 2px 8px; border: 1px solid var(--site-rule); border-radius: 4px; color: var(--ifm-color-primary);">v${packageJson.version}</span>`, position: 'right'},
        {href: repositoryUrl, label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Quickstart', to: '/docs/getting-started/quickstart'},
            {label: 'Installation', to: '/docs/getting-started/installation'},
            {label: 'Core Concepts', to: '/docs/getting-started/core-concepts'},
            {label: 'Deploying Models', to: '/docs/guides/deploying-models'},
          ],
        },
        {
          title: 'Guides & Ops',
          items: [
            {label: 'VRAM Calculation', to: '/docs/guides/vram-calculation'},
            {label: 'Traffic Routing & Proxy', to: '/docs/guides/traffic-routing'},
            {label: 'Prometheus Monitoring', to: '/docs/guides/monitoring'},
            {label: 'Systemd Service', to: '/docs/guides/systemd-service'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'CLI Commands', to: '/docs/reference/cli'},
            {label: 'REST API', to: '/docs/reference/api'},
            {label: 'Settings & Config', to: '/docs/reference/settings'},
            {label: 'Hardware Matrix', to: '/docs/reference/hardware-compatibility'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: repositoryUrl},
            {label: 'Architecture', to: '/docs/architecture'},
            {label: 'Security Model', to: '/docs/security'},
            {label: 'Changelog', to: '/docs/changelog'},
            {label: 'Contributing', to: '/docs/contributing'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Vulcanite & vLLM GPU Manager Contributors. Open source under the Apache-2.0 License. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'yaml', 'toml', 'docker'],
    },
  } satisfies ThemeConfig,
};

export default config;
