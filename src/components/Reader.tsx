"use client";

import { useEffect, useMemo, useState } from "react";
import {
  buildBackward,
  buildForward,
  chapterRange,
  type EditionKey,
  type EditionMeta,
  type EditionText,
  type Links,
  type Manifest,
  type MethodKey,
} from "@/core/types";
import { buildRows, paginate, type Row } from "@/core/rows";

const METHOD_LABEL: Record<MethodKey, string> = {
  embedding: "意味の近さ",
  gale_church: "文字数だけ",
  diagonal: "位置だけ",
};

const METHOD_NOTE: Record<MethodKey, string> = {
  embedding:
    "多言語の文埋め込みで、偶然の水準を超えて似ている組だけを繋ぐ。超えない文は繋がない。",
  gale_church:
    "Gale & Church (1993) の長さモデル。文字数の比だけを手がかりにする。",
  diagonal:
    "対照。文番号を比例配分するだけで、本文を一文字も見ない。これに勝てない手法は何も学んでいない。",
};

const ROWS_PER_PAGE = 6;

type Loaded = {
  manifest: Manifest;
  texts: Record<EditionKey, EditionText>;
  links: Links;
};

async function loadAll(): Promise<Loaded> {
  // **相対にしない。** `trailingSlash: true` なので画面は `/yomu/` に置かれ、
  // `./data` は `/yomu/data` に解決されて 404 になる。
  // 型検査もビルドも vitest も全部緑のまま通り、実ブラウザで開いて初めて出た。
  const base = "/data";
  const [manifest, de, en, ja, links] = await Promise.all([
    fetch(`${base}/manifest.json`).then((r) => r.json() as Promise<Manifest>),
    fetch(`${base}/text-de.json`).then((r) => r.json() as Promise<EditionText>),
    fetch(`${base}/text-en.json`).then((r) => r.json() as Promise<EditionText>),
    fetch(`${base}/text-ja.json`).then((r) => r.json() as Promise<EditionText>),
    fetch(`${base}/links.json`).then((r) => r.json() as Promise<Links>),
  ]);
  return { manifest, texts: { de, en, ja }, links };
}

export default function Reader() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chapter, setChapter] = useState(1);
  const [method, setMethod] = useState<MethodKey>("embedding");
  const [page, setPage] = useState(0);
  const [anchor, setAnchor] = useState<number | null>(null);

  useEffect(() => {
    loadAll().then(setData).catch((e) => setError(String(e)));
  }, []);

  const maps = useMemo(() => {
    if (!data) return null;
    const de_en = data.links.de_en[method] ?? [];
    const de_ja = data.links.de_ja[method] ?? [];
    return {
      en: buildForward(de_en),
      ja: buildForward(de_ja),
      enBack: buildBackward(de_en),
      jaBack: buildBackward(de_ja),
    };
  }, [data, method]);

  const pages = useMemo(() => {
    if (!data || !maps) return null;
    const meta = (k: EditionKey) => {
      const m = data.manifest.editions.find((e) => e.key === k);
      if (!m) throw new Error(`版 ${k} が manifest に無い`);
      return m;
    };
    const rows = buildRows(
      data.texts.de,
      data.texts.en,
      data.texts.ja,
      maps.enBack,
      maps.jaBack,
      {
        de: chapterRange(meta("de"), chapter),
        en: chapterRange(meta("en"), chapter),
        ja: chapterRange(meta("ja"), chapter),
      },
    );
    return paginate(rows, ROWS_PER_PAGE);
  }, [data, maps, chapter]);

  if (error) {
    return (
      <p className="note" style={{ padding: "30px 0" }}>
        データを読めませんでした: {error}
      </p>
    );
  }
  if (!data || !maps || !pages) {
    return (
      <p className="note" style={{ padding: "30px 0" }}>
        読み込んでいます…
      </p>
    );
  }

  const current = pages[Math.min(page, pages.length - 1)];
  const linkedEn = anchor === null ? new Set<number>() : new Set(maps.en.get(anchor) ?? []);
  const linkedJa = anchor === null ? new Set<number>() : new Set(maps.ja.get(anchor) ?? []);
  const crossed = data.manifest.cross_chapter_links;
  const metaOf = (key: EditionKey): EditionMeta =>
    data.manifest.editions.find((e) => e.key === key)!;

  const go = (p: number) => {
    setPage(Math.max(0, Math.min(p, pages.length - 1)));
    setAnchor(null);
  };

  return (
    <>
      <div className="controls">
        <fieldset>
          <legend>章</legend>
          {data.manifest.chapters.map((c) => (
            <button
              key={c}
              type="button"
              className="chip"
              aria-pressed={c === chapter}
              onClick={() => {
                setChapter(c);
                setPage(0);
                setAnchor(null);
              }}
            >
              第{c}章
            </button>
          ))}
        </fieldset>
        <fieldset>
          <legend>縫い方</legend>
          {data.manifest.methods.map((m) => (
            <button
              key={m}
              type="button"
              className="chip"
              aria-pressed={m === method}
              onClick={() => {
                setMethod(m);
                setAnchor(null);
              }}
            >
              {METHOD_LABEL[m] ?? m}
            </button>
          ))}
        </fieldset>
        <fieldset>
          <legend>頁</legend>
          <button type="button" className="chip" onClick={() => go(page - 1)}>
            ←
          </button>
          <span className="note">
            {Math.min(page, pages.length - 1) + 1} / {pages.length}
          </span>
          <button type="button" className="chip" onClick={() => go(page + 1)}>
            →
          </button>
        </fieldset>
      </div>

      <p className="note" style={{ padding: "12px 0 0" }}>
        {METHOD_NOTE[method]}　独語の文をクリックすると、その相手が光ります。
        {method === "embedding"
          ? "　この手法は章を一本もまたぎません(章は縫う側に渡していないので、これは答え合わせです)。"
          : `　この手法は章をまたぐ対応を 独→英 ${crossed.de_en[method]} 本 / 独→日 ${crossed.de_ja[method]} 本 持っています。`}
      </p>

      <div className="legend">
        <span>
          <span className="swatch" style={{ background: "var(--amber)" }} />
          選んだ文
        </span>
        <span>
          <span className="swatch" style={{ background: "rgba(224,178,90,0.26)" }} />
          その相手
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--bg-raised)" }} />
          相手のいない文
        </span>
        <span>行は独語の段落で揃えてある。</span>
      </div>

      <div className="rowhead">
        {(["de", "en", "ja"] as EditionKey[]).map((k) => (
          <div key={k} className="colhead" data-edition={k}>
            <b>{metaOf(k).title}</b>
            <span>
              {metaOf(k).translator ? `${metaOf(k).translator} 訳` : "原文"} ・{" "}
              {metaOf(k).sentences} 文
            </span>
          </div>
        ))}
      </div>

      {current.map((row) => (
        <RowView
          key={row.paragraph}
          row={row}
          texts={data.texts}
          anchor={anchor}
          linkedEn={linkedEn}
          linkedJa={linkedJa}
          hasPartner={{
            de: (i) => maps.en.has(i) || maps.ja.has(i),
            en: (j) => maps.enBack.has(j),
            ja: (j) => maps.jaBack.has(j),
          }}
          onPick={(i) => setAnchor((cur) => (cur === i ? null : i))}
        />
      ))}

      <div className="pager">
        <button type="button" className="chip" onClick={() => go(page - 1)}>
          ← 前の頁
        </button>
        <button type="button" className="chip" onClick={() => go(page + 1)}>
          次の頁 →
        </button>
      </div>
    </>
  );
}

function RowView({
  row,
  texts,
  anchor,
  linkedEn,
  linkedJa,
  hasPartner,
  onPick,
}: {
  row: Row;
  texts: Record<EditionKey, EditionText>;
  anchor: number | null;
  linkedEn: Set<number>;
  linkedJa: Set<number>;
  hasPartner: Record<EditionKey, (i: number) => boolean>;
  onPick: (i: number) => void;
}) {
  const cell = (
    key: EditionKey,
    idxs: number[],
    linked: Set<number> | null,
    clickable: boolean,
  ) => (
    <div className="cell" data-edition={key}>
      {idxs.length === 0 ? (
        <span className="empty">—</span>
      ) : (
        idxs.map((i) => {
          const role =
            clickable && anchor === i
              ? "anchor"
              : linked?.has(i)
                ? "linked"
                : hasPartner[key](i)
                  ? undefined
                  : "orphan";
          return (
            <span
              key={i}
              className="sentence"
              data-role={role}
              data-index={i}
              onClick={clickable ? () => onPick(i) : undefined}
              role={clickable ? "button" : undefined}
              tabIndex={clickable ? 0 : undefined}
              onKeyDown={
                clickable
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onPick(i);
                      }
                    }
                  : undefined
              }
            >
              {texts[key].sentences[i]}{" "}
            </span>
          );
        })
      )}
    </div>
  );

  return (
    <section className="row">
      {cell("de", row.de, null, true)}
      {cell("en", row.en, linkedEn, false)}
      {cell("ja", row.ja, linkedJa, false)}
    </section>
  );
}
