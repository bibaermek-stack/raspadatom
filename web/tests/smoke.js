/* Браузер смоук-тесті (Playwright, Chromium): беттің қатесіз жүктелуі және негізгі сценарийлер.
   node web/tests/smoke.js [index.html жолы]  -> соңғы жол JSON; қате болса шығу коды 1 */
'use strict';
const path = require('path');
const { chromium } = require('playwright');

const file = process.argv[2] || path.join(__dirname, '..', 'index.html');
const fail = [];
const check = (ok, msg) => { if (!ok) fail.push(msg); };

(async () => {
  const browser = await chromium.launch();
  for (const [name, vp] of [['desktop', { width: 1440, height: 1000 }], ['phone', { width: 390, height: 844 }]]) {
    const page = await (await browser.newContext({ viewport: vp })).newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    page.on('console', m => { if (m.type() === 'error' && !/fonts\.(googleapis|gstatic)|ERR_FAILED/.test(m.text())) errs.push(m.text()); });
    await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
    await page.goto('file://' + file);
    await page.waitForTimeout(700);

    check(await page.locator('#linesTable tbody tr').count() > 3, name + ': сызықтар кестесі бос');
    check(await page.locator('#aiBody .bar').count() > 3, name + ': ЖИ бағандары жоқ');
    // көлденең скролл болмауы керек
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    check(over <= 1, name + ': көлденең скролл ' + over + 'px');
    // кенептер бос емес
    const painted = await page.evaluate(() => ['strip', 'graph', 'levels', 'atomCanvas'].map(id => {
      const c = document.getElementById(id), x = c.getContext('2d'); const d = x.getImageData(0, 0, c.width, c.height).data;
      let n = 0; for (let i = 0; i < d.length; i += 4 * 97) if (d[i] + d[i + 1] + d[i + 2] > 40) n++;
      return [id, n];
    }));
    painted.forEach(([id, n]) => check(n > 15, name + ': ' + id + ' бос көрінеді (' + n + ')'));

    // сценарийлер: мысалдар, режимдер, 3D, жұмбақ қоспа
    const nPre = await page.locator('#presets .chip').count();
    for (let i = 0; i < nPre; i++) {
      await page.locator('#presets .chip').nth(i).click();
      await page.waitForTimeout(150);
    }
    await page.click('[data-scene=bench]'); await page.waitForTimeout(300);
    await page.click('[data-mode=emission]'); await page.click('[data-range=vis]'); await page.click('[data-scale=log]');
    await page.click('[data-scene=atom]'); await page.waitForTimeout(300);
    await page.locator('#linesTable tbody tr').nth(2).click();
    await page.click('#btnMystery'); await page.waitForTimeout(250);
    await page.click('.sp[data-n="Na I"]'); await page.click('#bCheck'); await page.waitForTimeout(150);
    await page.click('#bReveal'); await page.waitForTimeout(150);
    check(!(await page.locator('#bExit').isHidden()), name + ': Шығу түймесі жоқ');
    await page.click('#bExit'); await page.waitForTimeout(250);
    // барлық түр үшін өту тексеру: әр түрді таңдау қатесіз
    const names = await page.$$eval('#speciesGrid .sp', b => b.map(x => x.dataset.n));
    for (const n of names) { await page.click('.sp[data-n="' + n + '"]'); await page.waitForTimeout(20); }
    check(errs.length === 0, name + ': консоль қателері: ' + errs.slice(0, 3).join(' | '));
    await page.close();
  }
  await browser.close();
  console.log(JSON.stringify({ ok: fail.length === 0, fail }));
  process.exit(fail.length ? 1 : 0);
})();
