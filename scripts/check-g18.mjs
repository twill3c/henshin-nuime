/**
 * G-18 — 三本目の経路の照合。
 *
 * 経路A(onnxruntime / Python)と経路B(自前 NumPy)は L3 で突き合わせた。
 * ここは経路C —— **transformers.js が同じ ONNX を WASM で回す**。
 *
 * **隠し口を作らない。** 実装に `window.__test` のような穴を開けると、
 * 検品器が「利用者が触るもの」ではなく「検品器のために用意したもの」を見ることになる。
 * だから画面に**実際に表示される類似度**を読んで、局所で計算した期待値と比べる。
 * モデルが違えば、プーリングが違えば、切り詰め長が違えば、この値は 0.001 では
 * 済まない幅で動く。
 *
 * 期待値は `tests/fixtures/browser-embed.json`(局所で量子化モデルを回したもの)。
 *
 * **118 MB を Hugging Face から取りに行くので数分かかる。** 通常の検品
 * (`scripts/inspect.mjs`)とは別に置いてあるのはそのため。
 */

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const ROOT = join(process.cwd(), "out");
const FIXTURE = join(process.cwd(), "tests", "fixtures", "browser-embed.json");
const SCORE_TOLERANCE = 0.01; // 表示は小数 3 桁。実装の取り違えはこの幅を軽く超える
const LOAD_TIMEOUT = 15 * 60 * 1000;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".bin": "application/octet-stream",
  ".wasm": "application/wasm",
};

function serve() {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      let p = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname)));
      if (!p.startsWith(ROOT)) return res.writeHead(403).end();
      if (!extname(p)) p = join(p, "index.html");
      try {
        res.writeHead(200, {
          "content-type": MIME[extname(p)] ?? "application/octet-stream",
        });
        res.end(await readFile(p));
      } catch {
        res.writeHead(404).end("not found");
      }
    });
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

const problems = [];
const note = (m) => problems.push(m);

async function main() {
  if (!existsSync(ROOT)) {
    console.error("out/ が無い。`npm run build` を先に走らせること");
    process.exit(2);
  }
  if (!existsSync(FIXTURE)) {
    console.error(`${FIXTURE} が無い。期待値を先に作ること`);
    process.exit(2);
  }
  const fixture = JSON.parse(await readFile(FIXTURE, "utf-8"));
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  try {
    await page.goto(`${base}/nuu/`, { waitUntil: "networkidle" });

    // **押すまで通信しない**ことを先に確かめる(N-03)。
    const beforeClick = await page.evaluate(() =>
      performance.getEntriesByType("resource").filter((r) => r.name.includes("huggingface"))
        .length,
    );
    if (beforeClick !== 0) {
      note(`押す前に Hugging Face へ ${beforeClick} 件の要求が出ている`);
    }

    const button = page.getByRole("button", { name: /モデルを読み込む/ });
    if ((await button.count()) !== 1) {
      note("読み込みボタンが見つからない");
      throw new Error("読み込みボタンが見つからない");
    }
    console.log("モデルを読み込んでいます(118 MB・数分かかります)…");
    await button.click();
    await page.waitForSelector(".query", { timeout: LOAD_TIMEOUT });
    console.log("読み込み完了。照合に入ります。");

    for (const c of fixture.cases) {
      await page.fill(".query", c.text);
      await page.getByRole("button", { name: "縫う" }).click();
      await page.waitForSelector(".hits", { timeout: 120000 });
      // 前の結果を読まないよう、表示が入れ替わるまで待つ
      await page.waitForFunction(
        (expected) => {
          const first = document.querySelector('.hits .cell[data-edition="de"] p');
          return first && first.textContent && first.textContent.includes(expected);
        },
        c.expected.de.text.slice(0, 12),
        { timeout: 120000 },
      ).catch(() => {});

      const shown = await page.evaluate(() => {
        const out = {};
        for (const key of ["de", "en", "ja"]) {
          const p = document.querySelector(`.hits .cell[data-edition="${key}"] p`);
          if (!p) continue;
          const score = p.querySelector(".score")?.textContent ?? "";
          out[key] = {
            score: Number(score),
            text: (p.textContent ?? "").slice(score.length).trim(),
          };
        }
        return out;
      });

      for (const key of ["de", "en", "ja"]) {
        const want = c.expected[key];
        const got = shown[key];
        if (!got) {
          note(`${c.edition}#${c.index}: ${key} の結果が出ていない`);
          continue;
        }
        if (Math.abs(got.score - want.score) > SCORE_TOLERANCE) {
          note(
            `${c.edition}#${c.index} の ${key}: 類似度が 局所 ${want.score} 対 ` +
              `ブラウザ ${got.score}(許容 ${SCORE_TOLERANCE})`,
          );
        }
        if (got.text.slice(0, 24) !== want.text.slice(0, 24)) {
          note(
            `${c.edition}#${c.index} の ${key}: 一位の文が違う\n` +
              `      期待 ${want.text.slice(0, 40)}\n` +
              `      実際 ${got.text.slice(0, 40)}`,
          );
        }
      }
      console.log(`  ${c.edition}#${c.index} 照合済み`);
    }

    // 転送を実測する(N-03)。
    //
    // **`transferSize` は使えない。** クロスオリジンの資源は、相手が
    // `Timing-Allow-Origin` を返さない限り 0 を報告する。Hugging Face は返さないので、
    // 実際に 118 MB 落ちていても 0.0 MB と出る。最初その 0 を「モデルを読んでいない」
    // と読みかけた —— **測れない値と、値が 0 であることは違う**。
    // 代わりに要求の件数と、量子化モデルの名前が要求に現れたかを見る。
    const net = await page.evaluate(() => {
      const rs = performance.getEntriesByType("resource")
        .filter((r) => r.name.includes("huggingface"));
      return {
        count: rs.length,
        model: rs.filter((r) => /model_quantized\.onnx|\.onnx$/.test(r.name))
          .map((r) => r.name.split("/").slice(-1)[0]),
        transferReported: rs.reduce((a, r) => a + (r.transferSize || 0), 0),
      };
    });
    console.log(
      `Hugging Face への要求 ${net.count} 件(モデル: ${net.model.join(", ") || "なし"})` +
        `　転送量の報告は ${net.transferReported} バイト` +
        `(クロスオリジンなので 0 が返る。落ちていないという意味ではない)`,
    );
    if (net.count === 0) note("モデルを取りに行った形跡が無い");
    if (!net.model.some((m) => m.includes("quantized"))) {
      note(`量子化モデルを要求していない(要求されたモデル: ${net.model.join(", ") || "なし"})`);
    }

    if (errors.length) note(`ブラウザのエラー ${errors.length} 件: ${errors[0]}`);
  } finally {
    await browser.close();
    server.close();
  }

  if (problems.length) {
    console.error(`G-18 NG — ${problems.length} 件`);
    for (const p of problems) console.error("  " + p);
    process.exit(1);
  }
  console.log("G-18 OK — ブラウザ(transformers.js)の埋め込みが局所の量子化推論と一致");
}

await main();
