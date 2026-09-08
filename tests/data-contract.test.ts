/**
 * 焼いたデータと、公開面が読む型の契約(TEST_SPEC の T-057)。
 *
 * **境界をまたぐ契約は、テストできる場所に置く**(HC-190)。
 * `pipeline/bake.py` と `src/core/types.ts` は別の言語で書かれていて、
 * 片方だけを直しても誰も止めてくれない。ここが唯一の受け皿である。
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildBackward,
  buildForward,
  chapterRange,
  type EditionText,
  type Links,
  type Manifest,
} from "../src/core/types";

const DATA = join(process.cwd(), "public", "data");
/** 焼いた埋め込みの次元。`src/core/browser-embed.ts` の読み方と揃える。 */
const DIM = 384;
const read = <T,>(name: string): T =>
  JSON.parse(readFileSync(join(DATA, name), "utf-8")) as T;

const baked = existsSync(join(DATA, "manifest.json"));
const maybe = baked ? describe : describe.skip;

maybe("焼いたデータの契約", () => {
  const manifest = read<Manifest>("manifest.json");
  const links = read<Links>("links.json");
  const texts = {
    de: read<EditionText>("text-de.json"),
    en: read<EditionText>("text-en.json"),
    ja: read<EditionText>("text-ja.json"),
  };

  it("manifest の版が三つとも揃い、本文の長さと一致する", () => {
    expect(manifest.editions.map((e) => e.key)).toEqual(["de", "en", "ja"]);
    for (const e of manifest.editions) {
      const t = texts[e.key];
      expect(t.id).toBe(e.id);
      expect(t.sentences.length).toBe(e.sentences);
      expect(t.chapter.length).toBe(e.sentences);
      expect(t.paragraph.length).toBe(e.sentences);
    }
  });

  it("章の開始位置が本文の章番号と食い違わない", () => {
    for (const e of manifest.editions) {
      const t = texts[e.key];
      expect(e.chapter_offsets.length).toBe(manifest.chapters.length);
      for (const ch of manifest.chapters) {
        const [start, end] = chapterRange(e, ch);
        expect(end).toBeGreaterThan(start);
        // その区間の文がすべてその章に属すること
        for (let i = start; i < end; i += 1) expect(t.chapter[i]).toBe(ch);
      }
      // 章の区間が全体を覆いきること(取りこぼしを許さない)
      const covered = manifest.chapters
        .map((ch) => { const [s, e2] = chapterRange(e, ch); return e2 - s; })
        .reduce((a, b) => a + b, 0);
      expect(covered).toBe(e.sentences);
    }
  });

  it("段落番号が単調で、飛ばない", () => {
    for (const e of manifest.editions) {
      const p = texts[e.key].paragraph;
      let seen = -1;
      for (const v of p) {
        expect(v === seen || v === seen + 1).toBe(true);
        seen = Math.max(seen, v);
      }
      expect(seen).toBeGreaterThan(0);
    }
  });

  it("対応の番号が本文の範囲に収まる", () => {
    const sizes = { de_en: ["de", "en"], de_ja: ["de", "ja"] } as const;
    for (const pair of manifest.pairs) {
      const [srcKey, dstKey] = sizes[pair];
      for (const method of manifest.methods) {
        const rows = links[pair][method];
        expect(rows.length).toBeGreaterThan(0);
        for (const [i, j] of rows) {
          expect(i).toBeGreaterThanOrEqual(0);
          expect(i).toBeLessThan(texts[srcKey].sentences.length);
          expect(j).toBeGreaterThanOrEqual(0);
          expect(j).toBeLessThan(texts[dstKey].sentences.length);
        }
      }
    }
  });

  it("対応が単調である(縫い目が交差しない)", () => {
    // **単調性は対応の単位の性質であって、展開した組の並びの性質ではない。**
    // 2 対 2 の対応は 2×2 の格子状に組を生むので、(i,j) 順に並べた j は
    // 211,212,211,212 のように上下する —— それは交差ではない。
    // 正しい述語は「src の文番号ごとの最小 j も最大 j も、i について非減少」。
    for (const pair of manifest.pairs) {
      for (const method of manifest.methods) {
        const lo = new Map<number, number>();
        const hi = new Map<number, number>();
        for (const [i, j] of links[pair][method]) {
          lo.set(i, Math.min(lo.get(i) ?? j, j));
          hi.set(i, Math.max(hi.get(i) ?? j, j));
        }
        const srcs = [...lo.keys()].sort((a, b) => a - b);
        expect(srcs.length).toBeGreaterThan(0);
        for (let k = 1; k < srcs.length; k += 1) {
          expect(lo.get(srcs[k])!).toBeGreaterThanOrEqual(lo.get(srcs[k - 1])!);
          expect(hi.get(srcs[k])!).toBeGreaterThanOrEqual(hi.get(srcs[k - 1])!);
        }
        // 格子状の対応が実際に存在すること —— 上の緩和が必要だったことの確認
        if (pair === "de_en" && method === "gale_church") {
          const grid = srcs.filter((i) => hi.get(i)! > lo.get(i)!).length;
          expect(grid).toBeGreaterThan(0);
        }
      }
    }
  });

  it("章をまたぐ対応の件数が、対応そのものから数え直した値と一致する", () => {
    // **manifest の数字を信じない。** 焼いた対応から数え直して突き合わせる。
    const of = { de_en: ["de", "en"], de_ja: ["de", "ja"] } as const;
    for (const pair of manifest.pairs) {
      const [srcKey, dstKey] = of[pair];
      for (const method of manifest.methods) {
        const counted = links[pair][method].filter(
          ([i, j]) => texts[srcKey].chapter[i] !== texts[dstKey].chapter[j],
        ).length;
        expect(counted).toBe(manifest.cross_chapter_links[pair][method]);
      }
    }
  });

  it("埋め込みだけが章を一本もまたがない(手法ごとに違う)", () => {
    // **一手法の性質を全手法の話に広げない。** L8 で実際にやりかけた誤り。
    for (const pair of manifest.pairs) {
      expect(manifest.cross_chapter_links[pair].embedding).toBe(0);
    }
    const others = manifest.pairs.flatMap((p) => [
      manifest.cross_chapter_links[p].gale_church,
      manifest.cross_chapter_links[p].diagonal,
    ]);
    expect(Math.max(...others)).toBeGreaterThan(0);
  });

  it("対応表の往復が一致する", () => {
    const rows = links.de_en.embedding;
    const fwd = buildForward(rows);
    const back = buildBackward(rows);
    for (const [i, j] of rows) {
      expect(fwd.get(i)).toContain(j);
      expect(back.get(j)).toContain(i);
    }
    const fwdTotal = [...fwd.values()].reduce((a, v) => a + v.length, 0);
    expect(fwdTotal).toBe(rows.length);
  });

  it("本文に空の文が無い", () => {
    for (const e of manifest.editions) {
      for (const s of texts[e.key].sentences) expect(s.trim().length).toBeGreaterThan(0);
    }
  });

  // 焼いた埋め込みは**生の float32** で配る。JSON と違って構造が入っていないので、
  // 形を取り違えても誰も落ちない —— 次元がずれても、文の本数がずれても、
  // `Float32Array` には黙って載る。読める形かどうかを、ここで数える(T-064)。
  it("焼いた埋め込みの形が本文と食い違わない", () => {
    for (const e of manifest.editions) {
      const path = join(DATA, `emb-${e.key}.bin`);
      if (!existsSync(path)) continue; // モデルが手元に無ければ焼かれない
      const bytes = readFileSync(path).length;
      expect(bytes % (DIM * 4)).toBe(0);
      expect(bytes / (DIM * 4)).toBe(e.sentences);
    }
  });

  it("焼いた埋め込みが正規化されておらず、有限で、定数でない", () => {
    for (const e of manifest.editions) {
      const path = join(DATA, `emb-${e.key}.bin`);
      if (!existsSync(path)) continue;
      const buf = readFileSync(path);
      const v = new Float32Array(
        buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
      );
      let norm = 0;
      for (let k = 0; k < DIM; k += 1) norm += v[k] * v[k];
      norm = Math.sqrt(norm);
      // **L2 正規化はモデルに入っていない。** 焼く側で勝手に正規化すると
      // 1 に張り付き、公開面の cos の分母が意味を失う。
      expect(Math.abs(norm - 1)).toBeGreaterThan(0.01);
      expect(Number.isFinite(norm)).toBe(true);
      // 全部同じ値なら「読めているが中身が壊れている」— 数えないと気づけない
      const first = v[0];
      expect(v.slice(0, 1000).some((x) => x !== first)).toBe(true);
    }
  });

  it("権利の表示義務が manifest に載っている", () => {
    for (const e of manifest.editions) {
      expect(e.attribution.length).toBeGreaterThan(0);
      expect(e.source_url).toMatch(/^https:\/\//);
    }
    const ja = manifest.editions.find((e) => e.key === "ja")!;
    expect(ja.translator).toBe("原田義人");
    expect(ja.attribution.join(" ")).toContain("底本");
  });
});
