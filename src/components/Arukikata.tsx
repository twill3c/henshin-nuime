"use client";

// 画面「歩き方 / 設計図」(F-12)。
//
// この解剖台の資産は、うまくいった数字ではなく **落ちた目玉・測って取り下げた主張・
// 測り違えて直した経緯** のほうである。だからそれを隠さずに一枚にまとめる。
//
// **数字はここで打ち直さない。** 焼いたデータ(manifest / zure / attention)から読む ——
// 同じ数を文書と画面の二箇所に書けば、必ず片方だけが古びる。
// 打ち直しているのは「何を測ろうとしたか」という文だけである。

import { useEffect, useState } from "react";
import type { Manifest, Zure } from "@/core/types";

type Verdict = {
  attention: { links: number; paragraph_agreement: number; coverage_src: number };
  diagonal: { links: number; paragraph_agreement: number };
  difference: number;
  p_display: string;
  passed: boolean;
  threshold: number;
};

type Loaded = {
  manifest: Manifest;
  zure: Zure;
  verdict: Verdict | null;
};

export default function Arukikata() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/data/manifest.json").then((r) => r.json()),
      fetch("/data/zure.json").then((r) => r.json()),
      fetch("/data/attention.json")
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null),
    ])
      .then(([manifest, zure, attention]) =>
        setData({ manifest, zure, verdict: attention?.verdict ?? null }),
      )
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="note">データを読めませんでした: {error}</p>;
  if (!data) return <p className="note">読み込んでいます…</p>;

  const { manifest, zure, verdict } = data;
  const own = manifest.own_translation;
  const deEn = zure.pairs.de_en;
  const tr = zure.translators;

  return (
    <>
      <p className="lead">
        この解剖台が何を測り、何を測れなかったかの記録。
        <strong>落ちた判定も、取り下げた主張も、測り違えて直した経緯も、そのまま置いてある</strong> ——
        通った数字だけを並べると、何を信じてよいのかが読み手に分からなくなる。
      </p>

      <section>
        <h2>何を測っているのか</h2>
        <p>
          カフカ『変身』の三つの版を、<strong>どの文がどの文に対応するか</strong>で縫い合わせる。
          縫うのは四つの手法 —— 文字数だけ・位置だけ・多言語埋め込み・自作 Transformer の注意。
          どれが良いかではなく、<strong>対応そのものが機械に見えるか</strong>を測っている。
        </p>
        <div className="stat-row">
          {manifest.editions.map((e) => (
            <div className="stat" key={e.key}>
              <span className="stat-label">
                {e.lang === "de" ? "独語原文" : e.lang === "en" ? "英訳" : "和訳"}
                {e.translator ? `・${e.translator}` : ""}
              </span>
              <span className="stat-value">{e.sentences} 文</span>
              <span className="stat-sub">{e.title}</span>
            </div>
          ))}
        </div>
        <p className="note">
          <strong>採点表は本文の側にある。</strong> 独語と英語は段落構造が完全に一致する
          (どちらも 97 段落)。二つのファイルは互いを参照せずに作られているので、
          これは<strong>正解ラベルの要らないオラクル</strong>になる。
          日本語はもっと細かく割れているので、「一致する」ではなく
          「独語の段落の<strong>細分</strong>である」という弱い述語だけを使う。
        </p>
      </section>

      <section>
        <h2>循環の禁止 — 約束ではなく構造で守る</h2>
        <ul className="walk">
          <li>
            <strong>段落と章の境界は、縫い目を引く器に一度も渡さない。</strong>
            渡せば「境界を守る縫い目が引けた」は恒等式になる。
            アライナ(<code>pipeline/align.py</code>)はプロジェクト内のモジュールを
            一つも読み込まず、標準ライブラリだけで閉じている ——
            段落を知っている型に触れる道が、構文として存在しない。本文を読む入口も置いていない。
          </li>
          <li>
            <strong>自前の和訳は、原田訳を読まずに独語原文だけから作る。</strong>
            読めば「訳者差」を測る対照が、自分自身の写しになる。
            訳を書く器(<code>pipeline/translate.py</code>)には日本語版を読む口が無い。
            <br />
            <span className="note">
              ただし<strong>これは器の話であって、書く人の話ではない</strong>。
              「書いた人が原田訳を読まなかったこと」は検査では固定できない ——
              そこは手続きの約束にとどまる。検査で固定できることと、できないことを混ぜて書かない。
            </span>
          </li>
          <li>
            <strong>消せない漏れは、消せないと書く。</strong>
            文は段落をまたがないので、段落境界は文境界の部分集合である。
            この漏れを消す方法は無い(段落をまたぐ文分割を原文が許さない)。
          </li>
        </ul>
      </section>

      {verdict && (
        <section>
          <h2>事前登録した目玉と、その結果</h2>
          <p>
            作業を始める前にこう書いた ——
            <em>
              「一冊だけで学習した小さな Transformer は翻訳はできない。しかしその
              cross-attention が引く独↔英の対応は、位置だけの対角線を置換検定で有意に上回る」
            </em>
            。前半は成立し、<strong>後半は落ちた</strong>。
          </p>
          <div className="verdict" data-passed={verdict.passed}>
            <strong>
              {verdict.passed ? "通過" : "不通過"} —— 注意由来の段落一致率{" "}
              {verdict.attention.paragraph_agreement.toFixed(4)} 対 対角線{" "}
              {verdict.diagonal.paragraph_agreement.toFixed(4)}、差 +
              {verdict.difference.toFixed(4)}、p = {verdict.p_display}
              (閾値 p &lt; {verdict.threshold})。
            </strong>
            <br />
            <span className="note">
              落ちた予測を消していない。閾値も後から緩めていない。
              主画面は<strong>埋め込み由来の縫い目</strong>で成立させる、という
              撤退の仕方を先に決めてあったので、アプリは死ななかった。
            </span>
          </div>
        </section>
      )}

      <section>
        <h2>測って取り下げた主張</h2>
        <p className="note">
          どれも「良い結果が出たので書いた」のではなく、
          <strong>書いてあったものを測ったら支持されなかったので直した</strong>ものである。
        </p>
        <ul className="walk">
          <li>
            <strong>「埋め込みは長さモデルを上回る」——独↔英では言えない。</strong>{" "}
            段落一致率の差は +
            {deEn.tests.agreement_vs_gale_church?.difference.toFixed(4)}、p ={" "}
            {deEn.tests.agreement_vs_gale_church?.p}。1.0000 対 0.9949 は共通部分の
            数文の違いにすぎない。差が言えるのは<strong>位置だけの対角線に対してだけ</strong>
            (差 +{deEn.tests.agreement_vs_diagonal?.difference.toFixed(4)}、p ={" "}
            {deEn.tests.agreement_vs_diagonal?.p})。
            <br />
            <span className="note">
              勝ちが言えるのは日本語を含む組である。細分の破れの差は +
              {zure.pairs.de_ja.tests.refinement_vs_gale_church_block1?.difference.toFixed(4)}
              (p = {zure.pairs.de_ja.tests.refinement_vs_gale_church_block1?.p})。
            </span>
          </li>
          <li>
            <strong>三角整合の満点は、品質の証拠ではなかった。</strong>{" "}
            正解ラベルを使わないこの物差しで 1.0000 を取ったのは、
            段落一致率で最下位の「位置だけ」である
            (整合率{" "}
            {zure.triangle.methods.diagonal.rate.toFixed(4)} 対 埋め込み{" "}
            {zure.triangle.methods.embedding.rate.toFixed(4)})。
            比例写像の合成は比例写像なので<strong>自明に一致する</strong>。
            必要条件であって十分条件ではないことを、
            「いちばん悪い手法が満点を取る」という形で実測できた。
          </li>
          <li>
            <strong>「どの手法も章をまたがない」は、一手法だけ見て言いかけた。</strong>{" "}
            章をまたぐ対応は手法ごとに違う —— 埋め込み{" "}
            {manifest.cross_chapter_links.de_ja.embedding} 本に対し、
            文字数だけ {manifest.cross_chapter_links.de_ja.gale_church} 本、
            位置だけ {manifest.cross_chapter_links.de_ja.diagonal} 本(独→日)。
            焼く前の検査が、一般化しすぎを止めた。
          </li>
        </ul>
      </section>

      <section>
        <h2>測り違えて、直したこと</h2>
        <p className="note">
          <strong>もっともらしい数字は検査を素通りする。</strong>
          以下はどれも、検査が全部緑のまま通りかけたものである。
        </p>
        <ul className="walk">
          {tr && (
            <li>
              <strong>訳者差を、最初は「差なし」と出した。</strong>
              原田訳の字数を<strong>縫い目の文単位</strong>で独語段落に足したところ、
              段落あたりの字数比は二人ともほぼ同じに見えた。縫い目は偶然の水準を超えない
              対応を組まないので日本語側の 2 割強を飛ばし、原田訳だけが少なく出ていた。
              気づけたのは<strong>全体の比と段落ごとの中央値が合わなかったから</strong>である。
              段落単位で割り当て直すと、
              <strong>
                {tr.longer.of} 段落のうち {tr.longer.harada} 段落で原田訳のほうが長い
              </strong>
              (差 +{tr.test.difference.toFixed(4)}、p {tr.test.p})という逆の結論になった。
              <br />
              <span className="note">
                部分の集計が全体と合うかを見る —— それを検査にした。
              </span>
            </li>
          )}
          <li>
            <strong>切り詰めの表が、道具の既定値のせいで嘘になった。</strong>
            トークナイザの配布ファイルには切り詰め(128)と埋め草が焼き込まれていて、
            読み込んだだけで長さを測ると<strong>全部が 128 に揃う</strong>。
            そこから「上限を超えたものは 0 件」というもっともらしい嘘が出た。
          </li>
          <li>
            <strong>図のはみ出し検査が、正しく描かれているものを 6 件撃った。</strong>{" "}
            <code>getBBox()</code> が返すのは<strong>変換を適用する前</strong>の矩形なので、
            回転した軸ラベルでは回転前の位置と枠を比べることになる。
            直す前にスクリーンショットで実物を見たので、正しいものを壊さずに済んだ。
          </li>
          <li>
            <strong>量子化したモデルは、まとめて通すか一文ずつ通すかで出力が変わる。</strong>
            期待値をまとめて作り、ブラウザは一文ずつ回すので、照合が落ちた。
            同じ入力で fp32 はビット単位で同一、量子化だけがずれる。
            揃える規則に<strong>バッチの組み方</strong>が抜けていた。
          </li>
          <li>
            <strong>訳語の揺れが、検査を素通りして出荷された。</strong>
            訳語の一貫性を見る検査は<strong>登録簿にある語しか見ない</strong>ので、
            登録していない語の揺れは緑のまま通る。
            「検査がある」と「検査が届いている」は別で、後者は被覆でしか言えない。
            いまは登録簿 {own.terms} 語と、その被覆を数えて出している。
          </li>
        </ul>
      </section>

      <section>
        <h2>自前の和訳</h2>
        <p>
          独語原文からの和訳を{" "}
          <strong>
            {own.paragraphs[0]}/{own.paragraphs[1]} 段落
          </strong>
          ({own.chars.toLocaleString()} 字)作った。原田訳と併記するためのもので、
          <strong>原田訳を読まずに</strong>作っている。落丁検査は四本 ——
          段落を飛ばしていないか、固有名が落ちていないか、数が落ちていないか、
          そして<strong>段落の中で文が落ちていないか</strong>。
        </p>
        <p className="note">
          四本目は、前の三本に穴があったから足した ——
          固有名も数も含まない文を一つ落とすと、段落は残り、名前も数も揃うので、
          どれも黙ってしまう。原文との字数比で捕まえる。
          検査は五本とも<strong>壊すと落ちること</strong>を確かめてある。
        </p>
      </section>

      <section>
        <h2>作りの決まり</h2>
        <ul className="walk">
          <li>
            <strong>サーバ関数も cron も DB も持たない。</strong>
            縫い目も本文も、公開の前に静的ファイルへ焼き込んである。
            ブラウザはそれを読むだけで、課金の生じる推論は一つも走らない。
          </li>
          <li>
            <strong>ブラウザ内の推論はボタンを押すまで始まらない。</strong>
            「自分の文を縫う」で使うモデルは 112.8 MiB あるので、
            押されるまで一切取りに行かない。押したあとの計算もすべて手元で終わり、
            入力した文はどこにも送られない。
          </li>
          <li>
            <strong>三つの経路で同じモデルを回して突き合わせている。</strong>
            onnxruntime・自前の NumPy 順伝播・ブラウザの WASM。
            一致は「同じ実装だから」ではなく、別々に書いたものが同じ数を出すことで確かめる。
          </li>
        </ul>
      </section>

      <section>
        <h2>素材</h2>
        <ul className="walk">
          {manifest.editions.map((e) => (
            <li key={e.key}>
              <strong>{e.title}</strong>
              {e.translator ? `(${e.translator} 訳)` : ""} —{" "}
              <a href={e.source_url}>{e.source_url}</a>
              <br />
              <span className="note">{e.attribution.join(" / ")}</span>
            </li>
          ))}
        </ul>
        <p className="note">
          三点とも公有。表示義務のある事項は台帳(<code>data/rights.json</code>)を正本として、
          検査が原本の奥付と突き合わせている。
        </p>
      </section>
    </>
  );
}
