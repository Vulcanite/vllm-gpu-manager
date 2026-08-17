const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function capture() {
  const outputDir = path.join(__dirname, '..', 'static', 'img', 'screenshots', 'web');
  fs.mkdirSync(outputDir, { recursive: true });

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
    defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 2 }
  });

  const page = await browser.newPage();

  console.log('Navigating to http://127.0.0.1:9095...');
  await page.goto('http://127.0.0.1:9095/', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1000));

  // 1. Dashboard Overview
  console.log('Capturing Dashboard Overview...');
  await page.screenshot({ path: path.join(outputDir, '01-dashboard-overview.png'), fullPage: false });

  // 2. Deploy Server Modal / View
  console.log('Opening Deploy Server Modal...');
  const deployButtons = await page.$$('button');
  for (const btn of deployButtons) {
    const text = await page.evaluate(el => el.textContent, btn);
    if (text && text.includes('Deploy')) {
      await btn.click();
      await new Promise(r => setTimeout(r, 800));
      break;
    }
  }
  await page.screenshot({ path: path.join(outputDir, '02-deploy-server-modal.png'), fullPage: false });

  // 3. Models / Active Containers
  console.log('Navigating to Models / Active Containers...');
  await page.goto('http://127.0.0.1:9095/', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 500));
  const navLinks = await page.$$('nav a, a, button');
  for (const link of navLinks) {
    const text = await page.evaluate(el => el.textContent, link);
    if (text && (text.includes('Models') || text.includes('Active'))) {
      await link.click();
      await new Promise(r => setTimeout(r, 600));
      break;
    }
  }
  await page.screenshot({ path: path.join(outputDir, '03-active-models-and-logs.png'), fullPage: false });

  // 4. Cache & Download Browser
  console.log('Navigating to Cache...');
  const allElements = await page.$$('nav a, a, button, [role="tab"]');
  for (const el of allElements) {
    const text = await page.evaluate(e => e.textContent, el);
    if (text && (text.includes('Cache') || text.includes('Downloaded'))) {
      await el.click();
      await new Promise(r => setTimeout(r, 600));
      break;
    }
  }
  await page.screenshot({ path: path.join(outputDir, '04-huggingface-cache-browser.png'), fullPage: false });

  // 5. Proxy Gateway
  console.log('Navigating to Proxy Gateway...');
  const proxyLinks = await page.$$('nav a, a, button, [role="tab"]');
  for (const el of proxyLinks) {
    const text = await page.evaluate(e => e.textContent, el);
    if (text && (text.includes('Proxy') || text.includes('Gateway') || text.includes('Users'))) {
      await el.click();
      await new Promise(r => setTimeout(r, 600));
      break;
    }
  }
  await page.screenshot({ path: path.join(outputDir, '05-proxy-gateway-users.png'), fullPage: false });

  // 6. Settings
  console.log('Navigating to Settings...');
  const settingsLinks = await page.$$('nav a, a, button, [role="tab"]');
  for (const el of settingsLinks) {
    const text = await page.evaluate(e => e.textContent, el);
    if (text && text.includes('Settings')) {
      await el.click();
      await new Promise(r => setTimeout(r, 600));
      break;
    }
  }
  await page.screenshot({ path: path.join(outputDir, '06-settings-configuration.png'), fullPage: false });

  await browser.close();
  console.log('Web Console screenshots captured successfully!');
}

capture().catch(err => {
  console.error(err);
  process.exit(1);
});
