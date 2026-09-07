/**
 * 実ブラウザ検品。`out/` を配ってから本物の Chromium で開き、
 * **在存でなく幾何と到達を測る**(HC-138)。
 *
 *   幾何 —— 横に溢れていないか、縦に伸びすぎていないか、三列が潰れていないか
 *   到達 —— クリックが実際に届き、状態が変わったか(沈黙と「変化なし」を分ける)
 *
 * **検品器自身に陽性対照を置く**(HC-080)。異常なしを返したとき、その検品器が
 * 実際に異常を捕まえられることを、わざと壊した状態で一度確かめる。
 *
 * 使い方: node scripts/inspect.mjs [--shots]
 */

import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const ROOT = join(process.cwd(), "out");
const SHOTS = join(process.cwd(), "artifacts", "shots");
const WIDTHS = [1280, 900, 420]; // **一つの幅では見ない**(HC-078)
const MAX_PAGE_HEIGHT = 16000; // 列が潰れると縦に伸びる。代理指標(HC-078)

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
};

function serve() {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      const url = new URL(req.url, "http://localhost");
      let p = normalize(join(ROOT, decodeURIComponent(url.pathname)));
      if (!p.startsWith(ROOT)) {
        res.writeHead(403).end();
        return;
      }
      if (!extname(p)) p = join(p, "index.html");
      try {
        const body = await readFile(p);
        res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
        res.end(body);
      } catch {
        res.writeHead(404).end("not found");
      }
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

const problems = [];
const note = (m) => problems.push(m);

/**
 * 図の中身が `viewBox` に収まっているかを測る(G-10 / HC-159)。
 *
 * **ページの横溢れ検査は、図の内側の切れを見ない。** 軸ラベル・凡例・注記は
 * 描画領域の外に置かれるのが普通なので、`viewBox` を**データの範囲**から
 * 決めると静かに切り取られる。各要素の境界矩形を `viewBox` と比べる。
 * 文字の左サイドベアリングぶんのはみ出しなど、**幅によってしか現れない**
 * ものもあるので、幅ごとに測る。
 */
async function measureFigures(page, label) {
  const report = await page.evaluate(() => {
    const out = [];
    for (const svg of document.querySelectorAll(".figure svg")) {
      // **画面座標で比べる。** `getBBox()` は要素自身の座標系(変換を適用する前)
      // の矩形を返すので、`rotate(-90)` を掛けた軸ラベルでは回転前の位置と
      // viewBox を比べることになり、正しく描かれているものを落とす。
      // `getBoundingClientRect()` なら変換も入れ子も込みで「描かれる場所」が出る。
      const frame = svg.getBoundingClientRect();
      let checked = 0;
      const bad = [];
      for (const el of svg.querySelectorAll("text, path, rect, line, circle")) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;
        checked += 1;
        const eps = 1;
        if (
          r.left < frame.left - eps ||
          r.top < frame.top - eps ||
          r.right > frame.right + eps ||
          r.bottom > frame.bottom + eps
        ) {
          bad.push({
            tag: el.tagName,
            text: (el.textContent ?? "").slice(0, 20),
            x: +r.left.toFixed(1), y: +r.top.toFixed(1),
            w: +r.width.toFixed(1), h: +r.height.toFixed(1),
          });
        }
      }
      out.push({
        box: { w: +frame.width.toFixed(1), h: +frame.height.toFixed(1) },
        checked,
        bad,
      });
    }
    return out;
  });

  if (!report.length) note(`${label}: 図が一つも見つからない`);
  for (const [k, r] of report.entries()) {
    if (r.checked < 5) {
      note(`${label}: 図 ${k} の測れた要素が ${r.checked} 個 — 走査が働いていない`);
    }
    for (const b of r.bad) {
      note(
        `${label}: 図 ${k} の <${b.tag}>「${b.text}」が図の枠からはみ出す ` +
          `(${b.x},${b.y} ${b.w}×${b.h} / 枠 ${r.box.w}×${r.box.h})`,
      );
    }
  }
  return report;
}

/** 幾何を測る。要素が在るかではなく、どこにどう置かれているかを見る。 */
async function measure(page, label) {
  const geo = await page.evaluate(() => {
    const doc = document.documentElement;
    // 一行目の三つの升で列の幅を測る。**行ごとに測ると同じ幅を何度も数える。**
    const firstRow = document.querySelector(".row");
    const cols = firstRow
      ? [...firstRow.querySelectorAll(".cell")].map((el) => {
          const r = el.getBoundingClientRect();
          return { w: Math.round(r.width), h: Math.round(r.height), x: Math.round(r.x) };
        })
      : [];
    return {
      scrollWidth: doc.scrollWidth,
      clientWidth: doc.clientWidth,
      scrollHeight: doc.scrollHeight,
      columns: cols,
      rows: document.querySelectorAll(".row").length,
      sentences: document.querySelectorAll(".sentence").length,
    };
  });
  if (geo.scrollWidth > geo.clientWidth + 1) {
    note(`${label}: 横に溢れている(${geo.scrollWidth} > ${geo.clientWidth})`);
  }
  if (geo.scrollHeight > MAX_PAGE_HEIGHT) {
    note(`${label}: 縦に伸びすぎ(${geo.scrollHeight}px)— 列の潰れを疑う`);
  }
  for (const [i, c] of geo.columns.entries()) {
    if (c.w < 120) note(`${label}: 列 ${i} の幅が ${c.w}px しかない`);
  }
  return geo;
}

async function main() {
  if (!existsSync(ROOT)) {
    console.error("out/ が無い。`npm run build` を先に走らせること");
    process.exit(2);
  }
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  const shots = process.argv.includes("--shots");
  if (shots) await mkdir(SHOTS, { recursive: true });

  try {
    for (const width of WIDTHS) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
      });

      await page.goto(`${base}/`, { waitUntil: "networkidle" });
      await measure(page, `/ @${width}`);

      await page.goto(`${base}/yomu/`, { waitUntil: "networkidle" });
      // データを読み終えるまで待つ。**待たずに測ると「読み込んでいます…」を測る。**
      // 待てなかったときは例外を投げず、理由として記録する ——
      // 投げると他の幅の検品が走らないまま止まる。
      try {
        await page.waitForSelector(".sentence", { timeout: 20000 });
      } catch {
        const shown = await page.evaluate(() => document.body.innerText.slice(0, 120));
        note(`/yomu @${width}: 文が出ない。画面にあるのは「${shown.trim()}」`);
        await page.close();
        continue;
      }
      const geo = await measure(page, `/yomu @${width}`);
      if (geo.columns.length !== 3) {
        note(`/yomu @${width}: 一行の升が ${geo.columns.length} 個(3 個のはず)`);
      }
      if (geo.rows < 2) note(`/yomu @${width}: 行が ${geo.rows} 本しかない`);
      if (geo.sentences < 20) {
        note(`/yomu @${width}: 文が ${geo.sentences} 個しか出ていない`);
      }

      // --- 到達: クリックが届き、状態が変わったか ---
      //
      // **見た目の状態を測るなら、遷移の完了を待つこと。** 色は 90ms かけて
      // 変わるので、クリック直後に `getComputedStyle` を読むと**変わる前の値**が
      // 返る。ここで数えているのは属性なので時刻に影響されないが、
      // 色や大きさを足すときは待ちを入れる(L8 で実際に読み違えた)。
      const first = page.locator('.cell[data-edition="de"] .sentence').first();
      await first.scrollIntoViewIfNeeded();
      await first.click();
      await page.waitForTimeout(200);
      const after = await page.evaluate(() => ({
        anchors: document.querySelectorAll('.sentence[data-role="anchor"]').length,
        linked: document.querySelectorAll('.sentence[data-role="linked"]').length,
      }));
      if (after.anchors !== 1) {
        note(`/yomu @${width}: クリック後の選択が ${after.anchors} 個(1 個のはず)`);
      }
      if (after.linked < 1) {
        note(`/yomu @${width}: クリックしても相手が光らない(${after.linked} 個)`);
      }
      // **属性が付いただけでは光っていない。** 実際に色が変わったかを見る。
      const painted = await page.evaluate(() => {
        const a = document.querySelector('.sentence[data-role="anchor"]');
        const l = document.querySelector('.sentence[data-role="linked"]');
        const bg = (el) => (el ? getComputedStyle(el).backgroundColor : null);
        return { anchor: bg(a), linked: bg(l) };
      });
      const transparent = (c) => !c || c === "rgba(0, 0, 0, 0)" || c === "transparent";
      if (transparent(painted.anchor)) {
        note(`/yomu @${width}: 選んだ文に背景色が付いていない(${painted.anchor})`);
      }
      if (transparent(painted.linked)) {
        note(`/yomu @${width}: 相手の文に背景色が付いていない(${painted.linked})`);
      }
      if (painted.anchor === painted.linked) {
        note(`/yomu @${width}: 選んだ文と相手が同じ見た目 — どちらが起点か分からない`);
      }
      // **撮るならここ。** この後で手法を切り替えると選択が解けるので、
      // 後で撮ると「選んでいない画面」を「選んだ画面」として保存してしまう。
      if (shots) {
        await page.screenshot({ path: join(SHOTS, `yomu-${width}-selected.png`) });
      }

      // --- 手法の切り替えが実際に効くか ---
      const before = await page.evaluate(
        () => document.querySelectorAll('.sentence[data-role="linked"]').length,
      );
      await page.getByRole("button", { name: "位置だけ" }).click();
      const changed = await page.evaluate(
        () => document.querySelectorAll('.sentence[data-role="linked"]').length,
      );
      if (before === changed && before === 0) {
        note(`/yomu @${width}: 手法を切り替えても何も光らない`);
      }

      // --- 図の二枚 ---
      for (const [path, name] of [["/chizu/", "縫い目の地図"], ["/obi/", "注意の帯"]]) {
        await page.goto(`${base}${path}`, { waitUntil: "networkidle" });
        try {
          await page.waitForSelector(".figure svg .seam", { timeout: 20000 });
        } catch {
          const shown = await page.evaluate(() => document.body.innerText.slice(0, 120));
          note(`${path} @${width}: 図が出ない。画面にあるのは「${shown.trim()}」`);
          continue;
        }
        await measure(page, `${path} @${width}`);
        await measureFigures(page, `${path} @${width}`);
        // 経路が実際に描かれているか(空の d は描かれない)
        const drawn = await page.evaluate(() =>
          [...document.querySelectorAll(".figure svg .seam")].filter(
            (p) => (p.getAttribute("d") ?? "").length > 10,
          ).length,
        );
        if (drawn < 1) note(`${path} @${width}: 経路が一本も描かれていない`);
        if (shots) {
          await page.screenshot({ path: join(SHOTS, `${name}-${width}.png`) });
        }
      }

      if (errors.length) note(`@${width}: ブラウザのエラー ${errors.length} 件: ${errors[0]}`);
      if (shots) {
        // 素の状態(既定の手法・何も選んでいない)を撮り直す。
        await page.goto(`${base}/yomu/`, { waitUntil: "networkidle" });
        await page.waitForSelector(".sentence");
        await page.screenshot({ path: join(SHOTS, `yomu-${width}.png`) });
      }
      await page.close();
    }

    // --- 陽性対照: 検品器が実際に撃つか ---
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await page.goto(`${base}/yomu/`, { waitUntil: "networkidle" });
    await page.waitForSelector(".sentence");
    const mark = problems.length;
    await page.evaluate(() => {
      const el = document.querySelector(".cell");
      if (el) el.style.width = "4000px";
    });
    await measure(page, "陽性対照");
    if (problems.length === mark) {
      console.error("検品器が異常を捕まえられない — 検品器のほうを疑うこと");
      process.exit(3);
    }
    problems.length = mark;
    await page.close();

    // --- 陽性対照 その二: 図の枠からのはみ出しを捕まえるか ---
    const fig = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await fig.goto(`${base}/chizu/`, { waitUntil: "networkidle" });
    await fig.waitForSelector(".figure svg .seam");
    const clean = await measureFigures(fig, "陽性対照(図・素の状態)");
    if (problems.length !== mark) {
      console.error("素の図が既にはみ出している — 対照の前提が崩れている");
      process.exit(3);
    }
    if (!clean.length || clean[0].checked < 5) {
      console.error("図の走査が働いていない");
      process.exit(3);
    }
    await fig.evaluate(() => {
      const t = document.querySelector(".figure svg .axis text");
      if (t) t.setAttribute("transform", "translate(-400 -400)");
    });
    await measureFigures(fig, "陽性対照(図・ずらした状態)");
    if (problems.length === mark) {
      console.error("図をずらしても検出できない — 図の検査が働いていない");
      process.exit(3);
    }
    problems.length = mark; // 対照で足した分は本物ではないので戻す
    await fig.close();
  } finally {
    await browser.close();
    server.close();
  }

  if (problems.length) {
    console.error(`検品 NG — ${problems.length} 件`);
    for (const p of problems) console.error("  " + p);
    process.exit(1);
  }
  console.log(`検品 OK — 幅 ${WIDTHS.join("/")} で溢れ・潰れ・到達とも問題なし`);
}

await main();
