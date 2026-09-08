"use client";

// 画面「一語の変身」(F-10)。
//
// 訳語登録簿の一語ごとに、独語の文と、そこから**縫い目でたどった**英訳・原田訳、
// そしてこの企画の自前訳を並べる。
//
// **縫い目はここでも被験体である。** 訳語を比べるために縫い目を使っているので、
// 縫い目が外れていれば並ぶ文も外れる。だから「引けなかった」ときは黙って
// 空欄にせず、引けなかったと書く。
//
// **理由は登録簿に書いたものをそのまま出す。** 画面のために書き直さない ——
// 書き直せば、検査が見ている文字列と読者が読む文字列が別物になる。

import { useEffect, useMemo, useState } from "react";
import type { Manifest, Words, WordRow } from "@/core/types";

type Loaded = { words: Words; manifest: Manifest };

const KIND_LABEL: Record<string, string> = {
  name: "固有名",
  term: "訳語",
};

export default function Kotoba() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [kind, setKind] = useState<"all" | "name" | "term">("all");
  const [open, setOpen] = useState<string | null>("Ungeziefer");

  useEffect(() => {
    Promise.all([
      fetch("/data/words.json").then((r) => r.json()),
      fetch("/data/manifest.json").then((r) => r.json()),
    ])
      .then(([words, manifest]) => setData({ words, manifest }))
      .catch((e) => setError(String(e)));
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const all = data.words.terms;
    return kind === "all" ? all : all.filter((w) => w.kind === kind);
  }, [data, kind]);

  if (error) return <p className="note">データを読めませんでした: {error}</p>;
  if (!data) return <p className="note">読み込んでいます…</p>;

  const own = data.manifest.own_translation;
  const [done, total] = own.paragraphs;

  return (
    <>
      <p className="lead">
        独語の一語が、三つの日本語・英語にどう移ったか。
        <strong>訳語は登録簿で一つに固定してある</strong> ——
        同じ語を二通りに訳したら検査が落ちる。ここに出ている「なぜその訳語か」は、
        その登録簿に書いてある文そのままである。
      </p>

      <div className="controls">
        <fieldset>
          <legend>種類</legend>
          {(["all", "name", "term"] as const).map((k) => (
            <button
              key={k}
              type="button"
              className="chip"
              aria-pressed={kind === k}
              onClick={() => setKind(k)}
            >
              {k === "all" ? `すべて ${data.words.terms.length}` : KIND_LABEL[k]}
            </button>
          ))}
        </fieldset>
      </div>

      <p className="note">
        自前和訳は <strong>{done}/{total} 段落</strong>({own.chars.toLocaleString()} 字)。
        章別 {Object.entries(own.by_chapter).map(([c, v]) => `${c} 章 ${v[0]}/${v[1]}`).join(" / ")}。
        まだ訳していない章に初めて出る語は、自前訳の欄が空になる ——
        <strong>空欄は「訳が無い」であって「訳語が無い」ではない</strong>。
      </p>

      <ul className="words">
        {rows.map((w) => (
          <WordCard
            key={w.term}
            row={w}
            open={open === w.term}
            onToggle={() => setOpen(open === w.term ? null : w.term)}
          />
        ))}
      </ul>
    </>
  );
}

function WordCard({
  row,
  open,
  onToggle,
}: {
  row: WordRow;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <li className="word" data-kind={row.kind}>
      <button type="button" className="word-head" onClick={onToggle} aria-expanded={open}>
        <span className="word-de">{row.term}</span>
        <span className="word-arrow">→</span>
        <span className="word-ja">{row.term_ja}</span>
        <span className="word-meta">
          {row.count} 回・初出 {row.first}
        </span>
      </button>

      {open && (
        <div className="word-body">
          <p className="word-why">{row.why}</p>

          <div className="word-row">
            <h4 data-edition="de">独語原文</h4>
            <p>{row.de.text}</p>
          </div>

          <div className="word-row">
            <h4 data-edition="en">英訳 Wyllie(縫い目でたどった文)</h4>
            {row.en.length ? (
              row.en.map((s) => <p key={s.index}>{s.text}</p>
              )
            ) : (
              <p className="empty">縫い目が引けなかった</p>
            )}
          </div>

          <div className="word-row">
            <h4 data-edition="ja">和訳 原田義人(縫い目でたどった文)</h4>
            {row.ja.length ? (
              row.ja.map((s) => <p key={s.index}>{s.text}</p>)
            ) : (
              <p className="empty">縫い目が引けなかった</p>
            )}
          </div>

          <div className="word-row">
            <h4 data-edition="own">この企画の訳({row.own.paragraph} 段落)</h4>
            {row.own.text ? (
              <p>{row.own.text}</p>
            ) : (
              <p className="empty">この段落はまだ訳していない</p>
            )}
          </div>
        </div>
      )}
    </li>
  );
}
