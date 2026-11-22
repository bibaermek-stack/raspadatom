/* Браузер смоук-тесті (Playwright, Chromium): беттің қатесіз жүктелуі және негізгі сценарийлер.
   node web/tests/smoke.js [index.html жолы]  -> соңғы жол JSON; қате болса шығу коды 1 */
'use strict';
const path = require('path');
const fs = require('fs');
const os = require('os');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const file = process.argv[2] || path.join(__dirname, '..', 'index.html');
// камера сынағы үшін: PNG мысал және жалған веб-камера бейнесі (Y4M)
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'spectra-smoke-'));
const { writePNG } = require('./png.js');
const D = require('../data/lines.js'), PH = require('../js/physics.js'), CSIM = require('../js/camsim.js');
const simImg = CSIM.render(CSIM.SOURCES.find(s => s.id === 'ne'), D.species, {}, PH.mulberry32(21)).img;
const pngPath = path.join(tmp, 'ne.png'); writePNG(pngPath, simImg.data, simImg.width, simImg.height);
const y4mPath = path.join(tmp, 'ne.y4m');
require('child_process').execFileSync('node', [path.join(__dirname, 'make_y4m.js'), y4mPath, 'ne'], { stdio: 'ignore' });
const fail = [];
const check = (ok, msg) => { if (!ok) fail.push(msg); };

(async () => {
  const browser = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-file-for-fake-video-capture=' + y4mPath] });
  for (const [name, vp] of [['desktop', { width: 1440, height: 1000 }], ['phone', { width: 390, height: 844 }]]) {
    const page = await (await browser.newContext({ viewport: vp, permissions: ['camera'] })).newPage();
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
    // ---- камера бөлімі ----
    await page.click('[data-view=cam]'); await page.waitForTimeout(500);
    check(await page.locator('#camAiBody .bar').count() > 3, name + ': камера: ЖИ бағандары жоқ');
    const camPainted = await page.evaluate(() => ['camCanvas', 'camGraph'].map(id => {
      const c = document.getElementById(id), x = c.getContext('2d'); const d = x.getImageData(0, 0, c.width, c.height).data;
      let n = 0; for (let i = 0; i < d.length; i += 4 * 97) if (d[i] + d[i + 1] + d[i + 2] > 40) n++;
      return [id, n];
    }));
    camPainted.forEach(([id, n]) => check(n > 15, name + ': ' + id + ' бос (' + n + ')'));
    await page.click('#camAutoCal'); await page.waitForTimeout(300);
    check(/cfl/.test(await page.textContent('#camCalState')), name + ': CFL автокалибровкасы өтпеді: ' + await page.textContent('#camCalMsg'));
    if (name === 'desktop') {
      assert.ok(await page.locator('#camCalLines tr').count() >= 3);
      await page.getByText('Калибровка профильдері', { exact: true }).click();
      await page.fill('#camProfileName', 'CFL сынағы');
      await page.click('#camSaveProfile');
      assert.equal(await page.inputValue('#camProfiles'), 'CFL сынағы');
      await page.click('#camClearCal');
      assert.ok(await page.locator('#camCapture').isDisabled());
      await page.click('#camApplyProfile');
      assert.ok(await page.locator('#camCapture').isEnabled());
      assert.match(await page.textContent('#camProfileStatus'), /қолданылды/);
    }
    for (const i of [1, 2, 3, 8]) { await page.locator('#camSamples .chip').nth(i).click(); await page.waitForTimeout(250); }
    check(await page.locator('#camAiBody .bar').count() > 3, name + ': камера: үлгілерден кейін ЖИ жоқ');
    // қолмен ROI: суретте сызық сүйреу (автоматты табу сәтсіз болатын нақты фотолар үшін)
    const box = await page.locator('#camCanvas').boundingBox();
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.5); await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.52, { steps: 4 }); await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.55, { steps: 4 }); await page.mouse.up();
    await page.waitForTimeout(300);
    check(/ROI қолмен берілді/.test(await page.textContent('#camStatus')) && await page.isChecked('#camLock'), name + ': қолмен ROI істемеді: ' + await page.textContent('#camStatus'));
    check(await page.locator('#camAiBody .bar').count() > 3, name + ': қолмен ROI-дан кейін ЖИ жоқ');
    await page.setInputFiles('#camFile', pngPath); await page.waitForTimeout(500);
    check(/ne\.png/.test(await page.textContent('#camStatus')), name + ': PNG жүктелмеді: ' + await page.textContent('#camStatus'));
    await page.click('#camLive'); await page.waitForTimeout(2500);
    check(/Тікелей камера/.test(await page.textContent('#camStatus')), name + ': тікелей камера іске қосылмады: ' + await page.textContent('#camStatus'));
    await page.click('#camLive'); await page.waitForTimeout(200);
    await page.click('[data-view=lab]'); await page.waitForTimeout(300);
    if (name === 'desktop') {
      await page.click('#labCapture');
      await page.fill('#measurementName', 'Лаборатория үлгісі');
      await page.click('#measurementSave');
      const saved = await page.inputValue('#measurementHistory');
      assert.ok(saved);
      await page.locator('#measurementFile').setInputFiles({
        name: 'outside-range.csv', mimeType: 'text/csv',
        buffer: Buffer.from('wavelength_nm,intensity\n800,0\n810,1\n820,0\n')
      });
      await page.waitForFunction(() => document.querySelector('#measurementAnalysis').textContent.includes('Құрамы белгісіз'));
      await page.selectOption('#measurementCompare', saved);
      assert.match(await page.textContent('#measurementSummary'), /Үзік сызық.*Лаборатория үлгісі/);
      const download = page.waitForEvent('download');
      await page.click('#measurementExport');
      assert.match(fs.readFileSync(await (await download).path(), 'utf8'), /^wavelength_nm,intensity,peak/);
      await page.reload();
      await page.selectOption('#measurementHistory', saved);
      await page.click('#measurementOpen');
      assert.equal(await page.inputValue('#measurementName'), 'Лаборатория үлгісі');
      await page.click('#measurementRemove');
      assert.equal(await page.locator('#measurementHistory option').count(), 1);
    }
    check(errs.length === 0, name + ': консоль қателері: ' + errs.slice(0, 3).join(' | '));
    await page.close();
  }
  await browser.close();
  console.log(JSON.stringify({ ok: fail.length === 0, fail }));
  process.exit(fail.length ? 1 : 0);
})();
