"use client";

import { useEffect, useMemo, useState } from "react";
import {
  chapterRange,
  type EditionText,
  type Manifest,
} from "@/core/types";
import { PLOT, linear, paragraphStarts, pathPoints, polyline, ticks } from "@/core/figure";

const W = 760;
const H = 560;

type Attention = {
  pair: "de_en";
  shape: [number, number];
  cells: [number, number, number][];
  chance_share: number;
  links: [number, number][];
  verdict: {
    attention: { links: number; paragraph_agreement: number; coverage_src: number };
    diagonal: { links: number; paragraph_agreement: number };
    difference: number;
    p_value: number;
    p_display: string;
    passed: boolean;
    threshold: number;
  };
};

type Loaded = {
  manifest: Manifest;
  de: EditionText;
  en: EditionText;
  attention: Attention;
};

export default function AttentionBand() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chapter, setChapter] = useState(1);

  useEffect(() => {
    Promise.all([
      fetch("/data/manifest.json").then((r) => r.json() as Promise<Manifest>),
      fetch("/data/text-de.json").then((r) => r.json() as Promise<EditionText>),
      fetch("/data/text-en.json").then((r) => r.json() as Promise<EditionText>),
      fetch("/data/attention.json").then((r) => r.json() as Promise<Attention>),
    ])
      .then(([manifest, de, en, attention]) =>
        setData({ manifest, de, en, attention }),
      )
      .catch((e) => setError(String(e)));
  }, []);

  const figure = useMemo(() => {
    if (!data) return null;
    const metaOf = (k: "de" | "en") =>
      data.manifest.editions.find((e) => e.key === k)!;
    const [s0, s1] = chapterRange(metaOf("de"), chapter);
    const [d0, d1] = chapterRange(metaOf("en"), chapter);
    const n = s1 - s0;
    const m = d1 - d0;
    const sx = linear(n, PLOT.left, W - PLOT.right);
    const sy = linear(m, H - PLOT.bottom, PLOT.top);
    const cellW = (W - PLOT.left - PLOT.right) / Math.max(n, 1);
    const cellH = (H - PLOT.top - PLOT.bottom) / Math.max(m, 1);

    const cells = data.attention.cells
      .filter(([i, j]) => i >= s0 && i < s1 && j >= d0 && j < d1)
      .map(([i, j, v]) => ({ x: sx(i - s0), y: sy(j - d0), v }));

    const attnLinks = data.attention.links
      .filter(([i, j]) => i >= s0 && i < s1 && j >= d0 && j < d1)
      .map(([i, j]) => [i - s0, j - d0] as [number, number]);

    // 対照(対角線)は位置だけで決まるので、その場で引ける
    const diag: [number, number][] = [];
    for (let i = 0; i < n; i += 1) diag.push([i, Math.round((i * m) / n)]);

    return {
      n, m, sx, sy, cellW, cellH, cells,
      attnPath: polyline(pathPoints(attnLinks), sx, sy),
      diagPath: polyline(diag, sx, sy),
      attnCount: attnLinks.length,
      srcParas: paragraphStarts(data.de.paragraph, s0, s1),
      dstParas: paragraphStarts(data.en.paragraph, d0, d1),
      xTicks: ticks(n, 8),
      yTicks: ticks(m, 8),
    };
  }, [data, chapter]);

  if (error) return <p className="note">データを読めませんでした: {error}</p>;
  if (!data || !figure) return <p className="note">読み込んでいます…</p>;

  const v = data.attention.verdict;
  const maxShare = Math.max(...data.attention.cells.map((c) => c[2]), 1e-9);

  return (
    <>
      <div className="verdict" data-passed={v.passed}>
        <b>目玉は落ちた。</b> 事前に「一冊だけで学習した Transformer の cross-attention は、
        段落番号比例の対角線を置換検定で p&lt;{v.threshold} で上回る」と登録した。
        結果は 段落一致率 {v.attention.paragraph_agreement.toFixed(4)} 対{" "}
        {v.diagonal.paragraph_agreement.toFixed(4)}、
        差 {v.difference >= 0 ? "+" : ""}{v.difference.toFixed(4)}、
        <b> p = {v.p_display}</b>。閾値に届かない。
        主張は「段落境界を対角線より上手に守るとは言えない」に差し替えた。
        <strong>予測は消していない。</strong>
      </div>

      <div className="controls">
        <fieldset>
          <legend>章</legend>
          {data.manifest.chapters.map((c) => (
            <button key={c} type="button" className="chip" aria-pressed={c === chapter}
              onClick={() => setChapter(c)}>第{c}章</button>
          ))}
        </fieldset>
      </div>

      <p className="note" style={{ padding: "12px 0 0" }}>
        升の濃さは「英語の文が独語の文へ向けた注意のシェア」。
        <strong>値があるのは帯の中だけ</strong>です —— 学習の対を位置だけで切った窓で
        作ったので、窓の外へは注意が向きようがありません。
        つまりこの手法は、対照(対角線)より<strong>自由度が小さい</strong>。
        偶然の水準は窓の構造から決まる {data.attention.chance_share} です。
      </p>

      <div className="figure">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label="cross-attention の帯と、注意由来の経路・対角線の比較">
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

          <g className="band">
            {figure.cells.map((c, k) => (
              <rect key={k} x={c.x - figure.cellW / 2} y={c.y - figure.cellH / 2}
                width={Math.max(figure.cellW, 1)} height={Math.max(figure.cellH, 1)}
                opacity={Math.min(1, c.v / maxShare)} />
            ))}
          </g>

          <rect className="frame" x={PLOT.left} y={PLOT.top}
            width={W - PLOT.left - PLOT.right} height={H - PLOT.top - PLOT.bottom} />

          <g className="axis">
            {figure.xTicks.map((val) => (
              <text key={`xt${val}`} x={figure.sx(val)} y={H - PLOT.bottom + 16}
                textAnchor="middle">{val}</text>
            ))}
            {figure.yTicks.map((val) => (
              <text key={`yt${val}`} x={PLOT.left - 8} y={figure.sy(val) + 4}
                textAnchor="end">{val}</text>
            ))}
            <text x={(PLOT.left + W - PLOT.right) / 2} y={H - 10} textAnchor="middle">
              独語の文(第{chapter}章の {figure.n} 文)
            </text>
            <text x={14} y={(PLOT.top + H - PLOT.bottom) / 2} textAnchor="middle"
              transform={`rotate(-90 14 ${(PLOT.top + H - PLOT.bottom) / 2})`}>
              英語の文({figure.m} 文)
            </text>
          </g>

          <path className="seam control" d={figure.diagPath} stroke="var(--ja)" />
          <path className="seam" d={figure.attnPath} stroke="var(--amber)" />
        </svg>
      </div>

      <div className="legend">
        <span>
          <span className="swatch" style={{ background: "var(--amber)" }} />
          注意から引いた経路(対応 {figure.attnCount} 本)
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--ja)" }} />
          対照 = 位置だけの対角線
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--en)" }} />
          注意の帯(濃いほどシェアが大きい)
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--line)" }} />
          段落の切れ目
        </span>
      </div>
    </>
  );
}
