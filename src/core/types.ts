// 公開面が読むデータの形。pipeline/bake.py が焼いたものと**一対一で対応する**。
//
// 境界をまたぐ契約なので、片側だけを直すと黙ってずれる。
// tests/data-contract.test.ts が、実際に焼いたファイルがこの形であることを確かめる。

export type EditionKey = "de" | "en" | "ja";
export type PairKey = "de_en" | "de_ja";
export type MethodKey = "embedding" | "gale_church" | "diagonal";

export type EditionMeta = {
  key: EditionKey;
  id: string;
  lang: string;
  title: string;
  translator: string | null;
  source_url: string;
  attribution: string[];
  sentences: number;
  /** 章の開始位置(通し番号)。第1章は 0 から始まるので長さは章数と同じ。 */
  chapter_offsets: number[];
};

export type Manifest = {
  chapters: number[];
  methods: MethodKey[];
  pairs: PairKey[];
  /** 章をまたぐ対応の件数。**手法ごとに出す** —— 一つにまとめない。 */
  cross_chapter_links: Record<PairKey, Record<MethodKey, number>>;
  editions: EditionMeta[];
};

export type EditionText = {
  id: string;
  sentences: string[];
  /** 文ごとの章番号(1 始まり)。 */
  chapter: number[];
  /** 文ごとの通し段落番号(0 始まり)。表示の区切りに使う。 */
  paragraph: number[];
};

/** [src の文番号, dst の文番号] の組。どちらも版ごとの通し番号。 */
export type Link = [number, number];

export type Links = Record<PairKey, Record<MethodKey, Link[]>>;

/** ある版の文から、相手の版の文へ引ける対応表。ホバーの反応に使う。 */
export type Forward = Map<number, number[]>;

export function buildForward(links: Link[]): Forward {
  const out: Forward = new Map();
  for (const [i, j] of links) {
    const cur = out.get(i);
    if (cur) cur.push(j);
    else out.set(i, [j]);
  }
  return out;
}

export function buildBackward(links: Link[]): Forward {
  const out: Forward = new Map();
  for (const [i, j] of links) {
    const cur = out.get(j);
    if (cur) cur.push(i);
    else out.set(j, [i]);
  }
  return out;
}

/** 章 `chapter`(1 始まり)に属する文の [開始, 終了) を返す。 */
export function chapterRange(meta: EditionMeta, chapter: number): [number, number] {
  const idx = chapter - 1;
  if (idx < 0 || idx >= meta.chapter_offsets.length) {
    throw new Error(`章 ${chapter} は ${meta.id} に無い`);
  }
  const start = meta.chapter_offsets[idx];
  const end =
    idx + 1 < meta.chapter_offsets.length
      ? meta.chapter_offsets[idx + 1]
      : meta.sentences;
  return [start, end];
}
