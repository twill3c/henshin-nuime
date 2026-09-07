/** 画面の状態を実測して出すだけの道具。目視の印象を数で確かめるために使う。 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const ROOT = join(process.cwd(), "out");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".txt": "text/plain" };

const server = createServer(async (req, res) => {
  let p = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!p.startsWith(ROOT)) return res.writeHead(403).end();
  if (!extname(p)) p = join(p, "index.html");
  try {
    res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(await readFile(p));
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto(`${base}/yomu/`, { waitUntil: "networkidle" });
await page.waitForSelector(".sentence");

const snap = () =>
  page.evaluate(() => {
    const roles = {};
    const byEd = {};
    for (const el of document.querySelectorAll(".sentence")) {
      const r = el.dataset.role ?? "(なし)";
      roles[r] = (roles[r] ?? 0) + 1;
      const ed = el.closest(".cell")?.dataset.edition ?? "?";
      byEd[ed] = byEd[ed] ?? {};
      byEd[ed][r] = (byEd[ed][r] ?? 0) + 1;
    }
    const first = document.querySelector('.cell[data-edition="de"] .sentence');
    return {
      roles, byEd,
      firstDeRole: first?.dataset.role ?? "(なし)",
      firstDeColor: first ? getComputedStyle(first).color : null,
      firstDeBg: first ? getComputedStyle(first).backgroundColor : null,
    };
  });

console.log("--- 読み込み直後 ---");
console.log(JSON.stringify(await snap(), null, 1));

await page.locator('.cell[data-edition="de"] .sentence').first().click();
console.log("--- クリック直後(トランジション中)---");
console.log(JSON.stringify(await snap(), null, 1));
// **色は 90ms かけて変わる。** 直後に測ると変わる前の値が返る。
await page.waitForTimeout(300);
console.log("--- トランジション完了後 ---");
console.log(JSON.stringify(await snap(), null, 1));

await browser.close();
server.close();
