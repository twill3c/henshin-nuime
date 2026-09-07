// 三列を**段落で揃えて**並べるための組み立て。
//
// 素朴に三列をそのまま流すと、文の数が違う(独 696 / 英 771 / 日 1227)ので
// 下に行くほどずれて、隣り合っていない文が横に並ぶ。それでは比較にならない。
//
// そこで**独語の段落を行の単位**にし、英語と日本語の文をその行に割り当てる。
// 割り当ては縫い目に従う —— 相手のいる文はその相手の段落へ、
// 相手のいない文は**直前の割り当てを引き継ぐ**(単調なので前に戻らない)。
// こうすると、どの文もちょうど一度だけ現れる。

import type { EditionText, Forward } from "./types";

export type Row = {
  /** 独語の通し段落番号。行の識別子でもある。 */
  paragraph: number;
  de: number[];
  en: number[];
  ja: number[];
};

/**
 * `assign` は「相手の版の文番号 → 独語の段落番号」を返す。
 * 相手のいない文は直前の値を引き継ぐので、取りこぼしが出ない。
 */
export function assignToParagraphs(
  count: number,
  backward: Forward,
  deParagraph: number[],
  fallback: number,
): number[] {
  const out = new Array<number>(count);
  let last = fallback;
  for (let j = 0; j < count; j += 1) {
    const partners = backward.get(j);
    if (partners && partners.length) {
      // 複数の独語文に結ばれることがある。**最も早い段落**に寄せる
      // —— 後ろに寄せると、前の行が空のまま残ることがある。
      let best = deParagraph[partners[0]];
      for (const i of partners) best = Math.min(best, deParagraph[i]);
      last = best;
    }
    out[j] = last;
  }
  return out;
}

export function buildRows(
  de: EditionText,
  en: EditionText,
  ja: EditionText,
  enBack: Forward,
  jaBack: Forward,
  range: { de: [number, number]; en: [number, number]; ja: [number, number] },
): Row[] {
  const firstPara = de.paragraph[range.de[0]];
  const enAssign = assignToParagraphs(en.sentences.length, enBack, de.paragraph, firstPara);
  const jaAssign = assignToParagraphs(ja.sentences.length, jaBack, de.paragraph, firstPara);

  const rows = new Map<number, Row>();
  const row = (p: number): Row => {
    let r = rows.get(p);
    if (!r) {
      r = { paragraph: p, de: [], en: [], ja: [] };
      rows.set(p, r);
    }
    return r;
  };

  for (let i = range.de[0]; i < range.de[1]; i += 1) row(de.paragraph[i]).de.push(i);
  for (let j = range.en[0]; j < range.en[1]; j += 1) row(enAssign[j]).en.push(j);
  for (let j = range.ja[0]; j < range.ja[1]; j += 1) row(jaAssign[j]).ja.push(j);

  return [...rows.values()].sort((a, b) => a.paragraph - b.paragraph);
}

/** 行を `size` ずつの頁に切る。長い章をそのまま一枚に流さないため。 */
export function paginate(rows: Row[], size: number): Row[][] {
  if (size <= 0) throw new Error("頁の大きさは正でなければならない");
  const pages: Row[][] = [];
  for (let i = 0; i < rows.length; i += size) pages.push(rows.slice(i, i + size));
  return pages.length ? pages : [[]];
}
