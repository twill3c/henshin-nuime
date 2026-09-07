/**
 * 三列を段落で揃える組み立ての検査(TEST_SPEC の T-058)。
 *
 * **どの文もちょうど一度だけ現れること**が要である。相手のいない文を
 * 落としたり、二つの行に入れたりすると、読み手には気づけない形で本文が壊れる。
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { assignToParagraphs, buildRows, paginate } from "../src/core/rows";
import {
  buildBackward,
  chapterRange,
  type EditionText,
  type Links,
  type Manifest,
} from "../src/core/types";

describe("段落への割り当て", () => {
  it("相手のいない文は直前の割り当てを引き継ぐ", () => {
    // 文 0 は段落 0 に、文 2 は段落 1 に結ばれる。文 1 と 3 は相手なし。
    const back = new Map<number, number[]>([
      [0, [0]],
      [2, [5]],
    ]);
    const deParagraph = [0, 0, 0, 0, 0, 1];
    const got = assignToParagraphs(4, back, deParagraph, 0);
    expect(got).toEqual([0, 0, 1, 1]);
  });

  it("最初から相手がいなければ既定値に落ちる", () => {
    expect(assignToParagraphs(3, new Map(), [0], 7)).toEqual([7, 7, 7]);
  });

  it("複数の相手があれば最も早い段落に寄せる", () => {
    // 後ろに寄せると前の行が空のまま残ることがあるので、前に寄せる。
    const back = new Map<number, number[]>([[0, [3, 1, 2]]]);
    const deParagraph = [0, 4, 2, 9];
    expect(assignToParagraphs(1, back, deParagraph, 0)).toEqual([2]);
  });
});

describe("頁分け", () => {
  it("端数が出ても取りこぼさない", () => {
    const rows = Array.from({ length: 7 }, (_, i) => ({
      paragraph: i,
      de: [],
      en: [],
      ja: [],
    }));
    const pages = paginate(rows, 3);
    expect(pages.map((p) => p.length)).toEqual([3, 3, 1]);
    expect(pages.flat().length).toBe(rows.length);
  });

  it("行が無くても一頁は返す(空白を描く場所が要る)", () => {
    expect(paginate([], 3)).toEqual([[]]);
  });

  it("大きさが 0 以下なら落ちる", () => {
    expect(() => paginate([], 0)).toThrow();
  });
});

const DATA = join(process.cwd(), "public", "data");
const baked = existsSync(join(DATA, "manifest.json"));
const maybe = baked ? describe : describe.skip;

maybe("実データでの行の組み立て", () => {
  const read = <T,>(n: string): T =>
    JSON.parse(readFileSync(join(DATA, n), "utf-8")) as T;
  const manifest = read<Manifest>("manifest.json");
  const links = read<Links>("links.json");
  const texts = {
    de: read<EditionText>("text-de.json"),
    en: read<EditionText>("text-en.json"),
    ja: read<EditionText>("text-ja.json"),
  };
  const metaOf = (k: "de" | "en" | "ja") =>
    manifest.editions.find((e) => e.key === k)!;

  for (const method of ["embedding", "gale_church", "diagonal"] as const) {
    it(`${method}: どの文もちょうど一度だけ現れる`, () => {
      for (const ch of manifest.chapters) {
        const rows = buildRows(
          texts.de,
          texts.en,
          texts.ja,
          buildBackward(links.de_en[method]),
          buildBackward(links.de_ja[method]),
          {
            de: chapterRange(metaOf("de"), ch),
            en: chapterRange(metaOf("en"), ch),
            ja: chapterRange(metaOf("ja"), ch),
          },
        );
        for (const key of ["de", "en", "ja"] as const) {
          const [start, end] = chapterRange(metaOf(key), ch);
          const seen = rows.flatMap((r) => r[key]);
          expect(new Set(seen).size).toBe(seen.length); // 重複なし
          expect(seen.length).toBe(end - start); // 取りこぼしなし
          expect(Math.min(...seen)).toBe(start);
          expect(Math.max(...seen)).toBe(end - 1);
        }
        // 行は段落の順に並ぶ
        for (let i = 1; i < rows.length; i += 1) {
          expect(rows[i].paragraph).toBeGreaterThan(rows[i - 1].paragraph);
        }
        // どの行も少なくとも一つの升が埋まっている(空の行は描く意味が無い)
        expect(
          rows.every((r) => r.de.length + r.en.length + r.ja.length > 0),
        ).toBe(true);

        // **独語の無い行は、章をまたぐ対応があるときにだけ生じる。**
        // 英語や日本語の文が別の章の独語文に結ばれると、その段落の行に
        // 独語の文が入らない。行の形と対応の性質を、別々の道から突き合わせる。
        const deLess = rows.filter((r) => r.de.length === 0).length;
        const crosses =
          manifest.cross_chapter_links.de_en[method] +
          manifest.cross_chapter_links.de_ja[method];
        if (crosses === 0) expect(deLess).toBe(0);
      }
    });
  }

  it("独語の無い行が出るのは、章をまたぐ対応を持つ手法だけ", () => {
    // 上のケースは「またがない手法なら 0」しか言わない。逆向きも押さえる ——
    // **この対照が無いと、常に 0 を返す実装でも緑になる。**
    const deLessOf = (method: "embedding" | "gale_church" | "diagonal") => {
      let total = 0;
      for (const ch of manifest.chapters) {
        const rows = buildRows(
          texts.de,
          texts.en,
          texts.ja,
          buildBackward(links.de_en[method]),
          buildBackward(links.de_ja[method]),
          {
            de: chapterRange(metaOf("de"), ch),
            en: chapterRange(metaOf("en"), ch),
            ja: chapterRange(metaOf("ja"), ch),
          },
        );
        total += rows.filter((r) => r.de.length === 0).length;
      }
      return total;
    };
    expect(deLessOf("embedding")).toBe(0);
    expect(deLessOf("diagonal")).toBeGreaterThan(0);
  });

  it("意味の近さでは、第1段落の行に第2段落の日本語が混ざらない", () => {
    // **手法によって行の中身が変わることの確認**。混ざるかどうかは縫い目次第で、
    // 混ざらないのは埋め込みがそこを正しく縫えているからである。
    const rows = buildRows(
      texts.de,
      texts.en,
      texts.ja,
      buildBackward(links.de_en.embedding),
      buildBackward(links.de_ja.embedding),
      {
        de: chapterRange(metaOf("de"), 1),
        en: chapterRange(metaOf("en"), 1),
        ja: chapterRange(metaOf("ja"), 1),
      },
    );
    const first = rows[0];
    const paras = new Set(first.ja.map((j) => texts.ja.paragraph[j]));
    expect(paras.size).toBe(1);
  });
});
