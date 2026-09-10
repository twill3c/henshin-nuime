"use client";

// 画面「ずれの図録」(F-11)。
//
// **一つの数字にまとめない。** 分割・併合の形、段落一致率、細分の破れ、被覆、
// 三角整合、訳者差 —— どれも別のものを見ている。並べて出して読み手に比べさせる。
//
// **落ちた判定も、限界のある物差しも、そのまま載せる。** 三角整合で満点を取るのは
// 段落一致率で最下位の対角線である(比例写像の合成は比例写像なので自明に一致する)。
// それを隠すと、この物差しが品質の根拠に見えてしまう。

import { useEffect, useMemo, useState } from "react";
import type { MethodKey, PairKey, Zure } from "@/core/types";
import { PLOT, linear } from "@/core/figure";

const W = 520;
const H = 360;

const METHOD_LABEL: Record<MethodKey, string> = {
  embedding: "意味の近さ",
  gale_church: "文字数だけ",
  diagonal: "位置だけ",
};

const PAIR_LABEL: Record<PairKey, string> = {
  de_en: "独語 → 英語",
  de_ja: "独語 → 日本語",
};

const SHAPE_LABEL: Record<string, string> = {
  "1-1": "1 対 1",
  "1-2": "1 が 2 に割れる",
  "2-1": "2 が 1 に縮む",
  "2-2": "2 対 2",
  "1-0": "落ちる(訳が無い)",
  "0-1": "湧く(原文に無い)",
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function Zure() {
  const [data, setData] = useState<Zure | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/zure.json")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const scatter = useMemo(() => {
    if (!data?.translators) return null;
    const rows = data.translators.rows;
    const all = rows.flatMap((r) => [r.harada, r.own]);
    const lo = Math.min(...all) - 0.02;
    const hi = Math.max(...all) + 0.02;
    const n = Math.max(1, Math.round((hi - lo) * 1000));
    const sx = linear(n, PLOT.left, W - PLOT.right);
    const sy = linear(n, H - PLOT.bottom, PLOT.top);
    const at = (v: number) => Math.round((v - lo) * 1000);
    const ticks = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6].filter(
      (t) => t >= lo && t <= hi,
    );
    return {
      lo,
      hi,
      points: rows.map((r) => ({
        x: sx(at(r.own)),
        y: sy(at(r.harada)),
        label: r.paragraph,
        haradaLonger: r.harada > r.own,
      })),
      // 対角線(二人が同じ長さなら、点はこの線に乗る)
      diag: { x1: sx(0), y1: sy(0), x2: sx(n), y2: sy(n) },
      ticks: ticks.map((t) => ({ v: t, x: sx(at(t)), y: sy(at(t)) })),
    };
  }, [data]);

  if (error) return <p className="note">データを読めませんでした: {error}</p>;
  if (!data) return <p className="note">読み込んでいます…</p>;

  const tr = data.translators;

  return (
    <>
      <p className="lead">
        縫い目のずれを、いくつもの向きから測った図録。
        <strong>一つの数字にまとめていない</strong> ——
        どれも別のものを見ているので、並べたまま出す。
      </p>

      {tr && (
        <section>
          <h2>訳者差 — 同じ原文から、どれだけの字数になったか</h2>
          <p className="note">
            この企画には<strong>公有の第二英訳が無い</strong>ので、訳者差は日本語側で測ると決めてある。
            自前訳が全 97 段落そろって、はじめて測れるようになった。
          </p>

          <div className="stat-row">
            <Stat label="独語原文" value={`${tr.totals.de.toLocaleString()} 字`} />
            <Stat
              label="原田訳"
              value={`${tr.totals.harada.toLocaleString()} 字`}
              sub={`原文比 ${(tr.totals.harada / tr.totals.de).toFixed(4)}`}
            />
            <Stat
              label="自前訳"
              value={`${tr.totals.own.toLocaleString()} 字`}
              sub={`原文比 ${(tr.totals.own / tr.totals.de).toFixed(4)}`}
            />
          </div>

          <p className="verdict" data-passed="true">
            <strong>
              97 段落のうち {tr.longer.harada} 段落で、原田訳のほうが長い。
            </strong>{" "}
            段落あたりの字数比の差は平均 +{tr.test.difference.toFixed(4)}、
            対応のある置換検定で p {tr.test.p}。
            先頭 {tr.exact_head.n} 段落だけを全数列挙で確かめると
            差 +{tr.exact_head.difference.toFixed(4)}・p {tr.exact_head.p} で、別経路でも同じ向き。
            <br />
            <span className="note">
              <strong>原因は測っていない。</strong> 訳者の個性か、1960 年という時代か、
              機械が訳すと縮むのか —— この数字はそれを分けていない。
              また自前訳は段落を独語に合わせて作っているので、
              <strong>段落の切り方の差はここでは測れない</strong>(字数比だけが測れる)。
              原田訳の段落を独語段落へ割り当てるのに縫い目を使っているので、
              縫い目が外れれば割り当ても外れる。割り当てられなかったのは{" "}
              {tr.unassigned_chars.toLocaleString()} 字(
              {pct(tr.unassigned_chars / tr.totals.harada)})。
            </span>
          </p>

          {scatter && (
            <div className="figure">
              <svg viewBox={`0 0 ${W} ${H}`} role="img"
                   aria-label="段落ごとの字数比。横が自前訳、縦が原田訳。">
                <rect className="frame" x={PLOT.left} y={PLOT.top}
                      width={W - PLOT.left - PLOT.right}
                      height={H - PLOT.top - PLOT.bottom} />
                <g className="grid">
                  {scatter.ticks.map((t) => (
                    <g key={t.v}>
                      <line x1={t.x} y1={PLOT.top} x2={t.x} y2={H - PLOT.bottom} />
                      <line x1={PLOT.left} y1={t.y} x2={W - PLOT.right} y2={t.y} />
                    </g>
                  ))}
                </g>
                <line className="seam control"
                      x1={scatter.diag.x1} y1={scatter.diag.y1}
                      x2={scatter.diag.x2} y2={scatter.diag.y2}
                      stroke="var(--ink-quiet)" />
                <g>
                  {scatter.points.map((p) => (
                    <circle key={p.label} cx={p.x} cy={p.y} r={3}
                            fill={p.haradaLonger ? "var(--ja)" : "var(--amber)"}
                            opacity={0.75}>
                      <title>{p.label}</title>
                    </circle>
                  ))}
                </g>
                <g className="axis">
                  {scatter.ticks.map((t) => (
                    <text key={`x${t.v}`} x={t.x} y={H - PLOT.bottom + 16}
                          textAnchor="middle">{t.v.toFixed(2)}</text>
                  ))}
                  {scatter.ticks.map((t) => (
                    <text key={`y${t.v}`} x={PLOT.left - 8} y={t.y + 4}
                          textAnchor="end">{t.v.toFixed(2)}</text>
                  ))}
                  <text x={(PLOT.left + W - PLOT.right) / 2} y={H - 8}
                        textAnchor="middle">自前訳 / 独語原文(字数比)</text>
                  <text transform={`rotate(-90 14 ${(PLOT.top + H - PLOT.bottom) / 2})`}
                        x={14} y={(PLOT.top + H - PLOT.bottom) / 2}
                        textAnchor="middle">原田訳 / 独語原文</text>
                </g>
              </svg>
            </div>
          )}
          <p className="legend">
            <span><span className="swatch" style={{ background: "var(--ja)" }} />原田訳のほうが長い段落</span>
            <span><span className="swatch" style={{ background: "var(--amber)" }} />自前訳のほうが長い段落</span>
            <span>灰の線 = 二人が同じ長さになる位置</span>
          </p>
        </section>
      )}

      <section>
        <h2>分割と併合 — 縫い目がどんな形を作ったか</h2>
        <p className="note">
          `1 対 1` ばかりなら、それは対応というより並置である。
          手法ごとに形の分布が違うことが、手法の性格そのものを表す。
        </p>
        {(Object.keys(data.pairs) as PairKey[]).map((pair) => {
          const p = data.pairs[pair];
          const shapes = [
            ...new Set(
              Object.values(p.methods).flatMap((m) => Object.keys(m.shapes)),
            ),
          ].sort();
          return (
            <div key={pair} className="table-wrap">
              <h3>{PAIR_LABEL[pair]}(原文 {p.n_src} 文 → 訳 {p.n_dst} 文・伸縮率 {p.c.toFixed(4)})</h3>
              <table>
                <thead>
                  <tr>
                    <th>手法</th>
                    {shapes.map((s) => <th key={s}>{SHAPE_LABEL[s] ?? s}</th>)}
                    <th>被覆(原文/訳)</th>
                  </tr>
                </thead>
                <tbody>
                  {(Object.keys(p.methods) as MethodKey[]).map((m) => (
                    <tr key={m}>
                      <th scope="row">{METHOD_LABEL[m]}</th>
                      {shapes.map((s) => (
                        <td key={s}>{p.methods[m].shapes[s] ?? 0}</td>
                      ))}
                      <td>
                        {pct(p.methods[m].coverage_src)} / {pct(p.methods[m].coverage_dst)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </section>

      <section>
        <h2>段落を守れたか</h2>
        <p className="note">
          段落境界はアライナに一度も渡していない(G-03)ので、これは held-out の数字である。
          <strong>被覆が手法ごとに違うので、共通部分でも測り直す</strong> ——
          飛ばすほど破れが減って見える物差しを、そのまま並べてはいけない。
        </p>
        {(Object.keys(data.pairs) as PairKey[]).map((pair) => {
          const p = data.pairs[pair];
          return (
            <div key={pair} className="table-wrap">
              <h3>
                {PAIR_LABEL[pair]}(共通部分 原文 {p.common_src_sentences} 文 /
                訳 {p.common_dst_paragraphs} 段落)
              </h3>
              <table>
                <thead>
                  <tr>
                    <th>手法</th>
                    <th>段落一致率</th>
                    <th>同・共通部分</th>
                    <th>細分の破れ</th>
                    <th>同・共通部分</th>
                  </tr>
                </thead>
                <tbody>
                  {(Object.keys(p.methods) as MethodKey[]).map((m) => {
                    const e = p.methods[m];
                    return (
                      <tr key={m}>
                        <th scope="row">{METHOD_LABEL[m]}</th>
                        <td>{e.paragraph_agreement === null ? "—" : e.paragraph_agreement.toFixed(4)}</td>
                        <td>{e.paragraph_agreement_common === null ? "—" : e.paragraph_agreement_common.toFixed(4)}</td>
                        <td>{e.refinement_violations}/{e.refinement_paragraphs}</td>
                        <td>{e.refinement_violations_common}/{e.refinement_paragraphs_common}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
        <p className="note">
          段落一致率は<strong>独↔英でしか定義できない</strong>(両版の段落構造が一致するため)。
          日本語を含む組では細分の破れで代用するが、この数はアライナの誤りと
          「日本語は独語の段落の細分である」という仮説の誤りを混ぜて測っている。
          だから対照との差だけを読む。
        </p>
      </section>

      {data.extrapolation && (
        <section>
          <h2>外挿検証 — 別の本でも同じことが言えるか</h2>
          <p className="note">
            同じ三者(カフカ / Wyllie 訳 / 原田義人訳)による
            <strong>{data.extrapolation.work}</strong>で、同じ手続きを回した。
            <strong>規則も閾値も当てはめ直していない</strong> ——
            文分割の規則も長さモデルの分散も、『変身』で決めたまま使う。
            当てはめ直せば外挿にならない。
          </p>

          {!data.extrapolation.english_is_public_domain && (
            <p className="verdict">
              <strong>英訳は公有ではないので、本文を一字も配らない。</strong>{" "}
              『審判』の英訳(PG #7849)は冒頭に COPYRIGHTED と明記され、
              著作権者の許諾で収録されている作品である。『変身』の英訳が公有だったので
              同じだろうと想定していたが、<strong>権利は配布元ではなく作品ごとに決まる</strong>。
              ここに出ているのは段落数などの集計値だけで、本文も埋め込みも配っていない。
            </p>
          )}

          <div className="table-wrap">
            <h3>構造(章題と後書きを外して数えた)</h3>
            <table>
              <thead>
                <tr>
                  <th>版</th><th>段落</th><th>文</th><th>本文字数</th>
                </tr>
              </thead>
              <tbody>
                {(["de", "en", "ja"] as const).map((k) => {
                  const s = data.extrapolation!.structure[k];
                  return (
                    <tr key={k}>
                      <th scope="row">
                        {k === "de" ? "独語原文" : k === "en" ? "英訳(本文は配らない)" : "和訳 原田義人"}
                      </th>
                      <td>{s.paragraphs.toLocaleString()}</td>
                      <td>{s.sentences.toLocaleString()}</td>
                      <td>{s.chars.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="verdict">
            <strong>
              『変身』で背骨だったオラクルが、別の本では成り立たない。
            </strong>{" "}
            独 {data.extrapolation.structure.de.paragraphs} 段落に対し英{" "}
            {data.extrapolation.structure.en.paragraphs} 段落 ——
            『変身』では 97 対 97 で総数も章別も完全に一致し、
            「段落一致率」という物差しはその一致の上に立っていた。
            『審判』ではその物差しが<strong>そもそも定義できない</strong>。
            使えるのは「英の段落は独の段落の粗視化である」という弱い述語で、
            これは『変身』の日本語で使った述語と<strong>向きが逆</strong>である。
            <br />
            <span className="note">
              機構は分けていない —— 訳者が併合したのか、Wyllie の底本がこの 1925 年版と
              違ったのか、この数字は答えない。
            </span>
          </p>

          <div className="table-wrap">
            <h3>
              縫い目そのもの(独 → 日・共通部分{" "}
              {data.extrapolation.align.common_dst_paragraphs.toLocaleString()} 段落)
            </h3>
            <table>
              <thead>
                <tr>
                  <th>手法</th><th>対応</th><th>細分の破れ</th>
                  <th>同・共通部分</th><th>被覆(原文/訳)</th>
                </tr>
              </thead>
              <tbody>
                {(["embedding", "gale_church", "diagonal"] as const).map((m) => {
                  const e = data.extrapolation!.align[m];
                  return (
                    <tr key={m}>
                      <th scope="row">{METHOD_LABEL[m]}</th>
                      <td>{e.links.toLocaleString()}</td>
                      <td>{e.refinement_violations}/{e.refinement_paragraphs}</td>
                      <td>{e.refinement_violations_common}/{e.refinement_paragraphs_common}</td>
                      <td>{pct(e.coverage_src)} / {pct(e.coverage_dst)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="verdict" data-passed="true">
            <strong>手法についての結論は外挿した。</strong> 埋め込みは長さモデルを +
            {data.extrapolation.align.tests.refinement_vs_gale_church_block1?.difference.toFixed(4)}
            (p {data.extrapolation.align.tests.refinement_vs_gale_church_block1?.p})、
            対角線を +
            {data.extrapolation.align.tests.refinement_vs_diagonal_block1?.difference.toFixed(4)}
            (p {data.extrapolation.align.tests.refinement_vs_diagonal_block1?.p})上回る
            —— ブロック幅 1 / 5 / 10 のすべてで同じ向き。
            <br />
            <span className="note">
              <strong>ただし物差しの厳しさは作品で変わる。</strong>
              対照の破れ率は『変身』の 34% から 5〜6% に落ちている。
              手法が良くなったのではない ——『審判』は日本語の段落が独語の 10.7 倍に
              細分されており(『変身』は 1.70 倍)、一段落あたりの文が少ないぶん
              またぐ機会そのものが減る。<strong>二つの本の数字を直接比べてはいけない。</strong>
              読めるのは、それぞれの本の中での対照との差だけである。
            </span>
          </p>
        </section>
      )}

      <section>
        <h2>三角整合 — 正解ラベルを使わない物差し</h2>
        <p className="note">
          <code>独→日</code> と <code>(独→英)∘(英→日)</code> が同じ文を指すか。
          正解ラベルは要らない。
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>手法</th>
                <th>整合率</th>
                <th>比べられた文</th>
                <th>片側だけ(直接 / 合成)</th>
                <th>帰無(中央)</th>
              </tr>
            </thead>
            <tbody>
              {(Object.keys(data.triangle.methods) as MethodKey[]).map((m) => {
                const t = data.triangle.methods[m];
                const nulls = data.triangle.null[m];
                const mid = nulls[Math.floor(nulls.length / 2)];
                return (
                  <tr key={m}>
                    <th scope="row">{METHOD_LABEL[m]}</th>
                    <td>{t.rate.toFixed(4)}</td>
                    <td>{t.compared}</td>
                    <td>{t.only_direct} / {t.only_composed}</td>
                    <td>{mid.toFixed(4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="verdict">
          <strong>満点を取ったのは、段落一致率で最下位の「位置だけ」である。</strong>
          比例写像の合成は比例写像なので、<strong>自明に一致する</strong>。
          この物差しは必要条件であって十分条件ではない ——
          それを「いちばん悪い手法が満点を取る」という形で実測できた。
          整合率だけを見て良し悪しを言ってはいけない。
        </p>
      </section>
    </>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}
