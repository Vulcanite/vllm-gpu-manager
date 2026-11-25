import type {Config} from '@docusaurus/types';
import type {Options, ThemeConfig} from '@docusaurus/preset-classic';
import {themes as prismThemes} from 'prism-react-renderer';
import packageJson from './package.json';

const organizationName = process.env.GITHUB_REPOSITORY_OWNER ?? 'Vulcanite';
const projectName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? 'vllm-gpu-manager';
const repositoryUrl = `https://github.com/${organizationName}/${projectName}`;
const editBranch = process.env.DOCS_EDIT_BRANCH ?? 'main';
const isUserSite = projectName === `${organizationName}.github.io`;
// CI passes the latest release tag; package.json is only a local fallback.
const docsVersion = (process.env.DOCS_VERSION || `v${packageJson.version}`).trim();

const config: Config = {
  title: 'vLLM GPU Manager',
  tagline: 'Run vLLM on your own GPU server.',
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
        content: 'Documentation for vLLM GPU Manager: one binary with a CLI and web dashboard to run vLLM in Docker on NVIDIA GPUs, with VRAM checks and an OpenAI-compatible proxy.',
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
        {to: '/docs/quickstart', label: 'Get Started', position: 'left'},
        {to: '/docs/guides/deploy', label: 'Guides', position: 'left'},
        {to: '/docs/reference/cli', label: 'Reference', position: 'left'},
        {type: 'html', value: `<span style="font-family: var(--site-mono); font-size: 0.8rem; padding: 2px 8px; border: 1px solid var(--site-rule); border-radius: 4px; color: var(--ifm-color-primary);">${docsVersion}</span>`, position: 'right'},
        {href: repositoryUrl, label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Install', to: '/docs/install'},
            {label: 'Quickstart', to: '/docs/quickstart'},
            {label: 'Deploy models', to: '/docs/guides/deploy'},
            {label: 'Proxy & routing', to: '/docs/guides/proxy'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'CLI', to: '/docs/reference/cli'},
            {label: 'REST API', to: '/docs/reference/api'},
            {label: 'Settings & environment', to: '/docs/reference/settings'},
            {label: 'Security', to: '/docs/security'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: repositoryUrl},
            {label: 'Why I built it', href: 'https://amishtp.tech/blog/vllm-gpu-manager'},
            {label: 'Changelog', to: '/docs/changelog'},
            {label: 'Contributing', to: '/docs/contributing'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Vulcanite and contributors. Apache-2.0 License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'yaml', 'toml', 'docker'],
    },
  } satisfies ThemeConfig,
};

export default config;
