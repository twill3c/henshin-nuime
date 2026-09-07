"use client";

import { useEffect, useMemo, useState } from "react";
import {
  chapterRange,
  type EditionKey,
  type EditionText,
  type Links,
  type Manifest,
  type MethodKey,
  type PairKey,
} from "@/core/types";
import { PLOT, linear, paragraphStarts, pathPoints, polyline, ticks } from "@/core/figure";

const W = 760;
const H = 560;

const METHOD_LABEL: Record<MethodKey, string> = {
  embedding: "意味の近さ",
  gale_church: "文字数だけ",
  diagonal: "位置だけ",
};
const METHOD_COLOR: Record<MethodKey, string> = {
  embedding: "var(--amber)",
  gale_church: "var(--en)",
  diagonal: "var(--ja)",
};

const PAIR_LABEL: Record<PairKey, [string, EditionKey, EditionKey]> = {
  de_en: ["独語 → 英語", "de", "en"],
  de_ja: ["独語 → 日本語", "de", "ja"],
};

type Loaded = {
  manifest: Manifest;
  texts: Record<EditionKey, EditionText>;
  links: Links;
};

export default function SeamMap() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chapter, setChapter] = useState(1);
  const [pair, setPair] = useState<PairKey>("de_ja");
  const [shown, setShown] = useState<MethodKey[]>([
    "embedding",
    "gale_church",
    "diagonal",
  ]);

  useEffect(() => {
    Promise.all([
      fetch("/data/manifest.json").then((r) => r.json()),
      fetch("/data/text-de.json").then((r) => r.json()),
      fetch("/data/text-en.json").then((r) => r.json()),
      fetch("/data/text-ja.json").then((r) => r.json()),
      fetch("/data/links.json").then((r) => r.json()),
    ])
      .then(([manifest, de, en, ja, links]) =>
        setData({ manifest, texts: { de, en, ja }, links }),
      )
      .catch((e) => setError(String(e)));
  }, []);

  const figure = useMemo(() => {
    if (!data) return null;
    const [, srcKey, dstKey] = PAIR_LABEL[pair];
    const metaOf = (k: EditionKey) =>
      data.manifest.editions.find((e) => e.key === k)!;
    const [s0, s1] = chapterRange(metaOf(srcKey), chapter);
    const [d0, d1] = chapterRange(metaOf(dstKey), chapter);
    const n = s1 - s0;
    const m = d1 - d0;
    const sx = linear(n, PLOT.left, W - PLOT.right);
    const sy = linear(m, H - PLOT.bottom, PLOT.top);

    const paths = shown.map((method) => {
      const local = data.links[pair][method]
        .filter(([i, j]) => i >= s0 && i < s1 && j >= d0 && j < d1)
        .map(([i, j]) => [i - s0, j - d0] as [number, number]);
      return { method, d: polyline(pathPoints(local), sx, sy), count: local.length };
    });

    return {
      n,
      m,
      sx,
      sy,
      paths,
      srcKey,
      dstKey,
      srcParas: paragraphStarts(data.texts[srcKey].paragraph, s0, s1),
      dstParas: paragraphStarts(data.texts[dstKey].paragraph, d0, d1),
      xTicks: ticks(n, 8),
      yTicks: ticks(m, 8),
    };
  }, [data, pair, chapter, shown]);

  if (error) return <p className="note">データを読めませんでした: {error}</p>;
  if (!data || !figure) return <p className="note">読み込んでいます…</p>;

  const toggle = (m: MethodKey) =>
    setShown((cur) =>
      cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m],
    );

  return (
    <>
      <div className="controls">
        <fieldset>
          <legend>章</legend>
          {data.manifest.chapters.map((c) => (
            <button key={c} type="button" className="chip" aria-pressed={c === chapter}
              onClick={() => setChapter(c)}>第{c}章</button>
          ))}
        </fieldset>
        <fieldset>
          <legend>組</legend>
          {(Object.keys(PAIR_LABEL) as PairKey[]).map((p) => (
            <button key={p} type="button" className="chip" aria-pressed={p === pair}
              onClick={() => setPair(p)}>{PAIR_LABEL[p][0]}</button>
          ))}
        </fieldset>
        <fieldset>
          <legend>重ねる</legend>
          {data.manifest.methods.map((m) => (
            <button key={m} type="button" className="chip" aria-pressed={shown.includes(m)}
              onClick={() => toggle(m)}>{METHOD_LABEL[m]}</button>
          ))}
        </fieldset>
      </div>

      <p className="note" style={{ padding: "12px 0 0" }}>
        横軸が独語の文、縦軸が相手の文。<strong>薄い格子は段落の切れ目</strong>で、
        これが答え合わせに使うオラクルそのものです(縫う側には一度も渡していません)。
        経路が格子の升目の中を通っていれば、その手法は段落を守れています。
      </p>

      <div className="figure">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label={`${PAIR_LABEL[pair][0]} の縫い目の経路。段落の切れ目を薄い格子で示す。`}>
          {/* 段落の格子。**オラクルを目に見える形にする** */}
          <g className="grid">
            {figure.srcParas.map((i) => (
              <line key={`v${i}`} x1={figure.sx(i)} y1={PLOT.top}
                x2={figure.sx(i)} y2={H - PLOT.bottom} />
            ))}
            {figure.dstParas.map((j) => (
              <line key={`h${j}`} x1={PLOT.left} y1={figure.sy(j)}
                x2={W - PLOT.right} y2={figure.sy(j)} />
            ))}
          </g>

          {/* 枠 */}
          <rect className="frame" x={PLOT.left} y={PLOT.top}
            width={W - PLOT.left - PLOT.right} height={H - PLOT.top - PLOT.bottom} />

          {/* 目盛り。位置も内容も同じデータから導く */}
          <g className="axis">
            {figure.xTicks.map((v) => (
              <text key={`xt${v}`} x={figure.sx(v)} y={H - PLOT.bottom + 16}
                textAnchor="middle">{v}</text>
            ))}
            {figure.yTicks.map((v) => (
              <text key={`yt${v}`} x={PLOT.left - 8} y={figure.sy(v) + 4}
                textAnchor="end">{v}</text>
            ))}
            <text x={(PLOT.left + W - PLOT.right) / 2} y={H - 10} textAnchor="middle">
              独語の文(第{chapter}章の {figure.n} 文)
            </text>
            <text x={14} y={(PLOT.top + H - PLOT.bottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90 14 ${(PLOT.top + H - PLOT.bottom) / 2})`}>
              相手の文({figure.m} 文)
            </text>
          </g>

          {figure.paths.map((p) => (
            <path key={p.method} className="seam" d={p.d}
              stroke={METHOD_COLOR[p.method]} />
          ))}
        </svg>
      </div>

      <div className="legend">
        {figure.paths.map((p) => (
          <span key={p.method}>
            <span className="swatch" style={{ background: METHOD_COLOR[p.method] }} />
            {METHOD_LABEL[p.method]}(対応 {p.count} 本)
          </span>
        ))}
        <span>
          <span className="swatch" style={{ background: "var(--line)" }} />
          段落の切れ目
        </span>
      </div>
    </>
  );
}
