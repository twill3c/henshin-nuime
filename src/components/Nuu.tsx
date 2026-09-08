"use client";

import { useEffect, useState } from "react";
import {
  MODEL_BYTES,
  MODEL_ID,
  fetchEmbeddings,
  loadEmbedder,
  nearest,
  type Embedder,
} from "@/core/browser-embed";
import type { EditionKey, EditionText, Manifest } from "@/core/types";

const KEYS: EditionKey[] = ["de", "en", "ja"];
const LABEL: Record<EditionKey, string> = {
  de: "独語原文",
  en: "英訳 Wyllie",
  ja: "和訳 原田義人",
};

type Corpus = {
  manifest: Manifest;
  texts: Record<EditionKey, EditionText>;
  vecs: Record<EditionKey, { dim: number; vectors: Float32Array }>;
};

type Hit = { key: EditionKey; index: number; score: number };

export default function Nuu() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [stage, setStage] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [embedder, setEmbedder] = useState<Embedder | null>(null);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/data/manifest.json")
      .then((r) => r.json())
      .then(setManifest)
      .catch((e) => setMessage(String(e)));
  }, []);

  const start = async () => {
    setStage("loading");
    setMessage("モデルと本文を取りに行っています…");
    try {
      const [fn, de, en, ja, vde, ven, vja] = await Promise.all([
        loadEmbedder((f, file) => {
          setProgress(f);
          if (file) setMessage(`${file} を読み込み中`);
        }),
        fetch("/data/text-de.json").then((r) => r.json()),
        fetch("/data/text-en.json").then((r) => r.json()),
        fetch("/data/text-ja.json").then((r) => r.json()),
        fetchEmbeddings("de"),
        fetchEmbeddings("en"),
        fetchEmbeddings("ja"),
      ]);
      setEmbedder(() => fn);
      setCorpus({
        manifest: manifest!,
        texts: { de, en, ja },
        vecs: { de: vde, en: ven, ja: vja },
      });
      setStage("ready");
      setMessage("");
    } catch (e) {
      setStage("error");
      setMessage(String(e));
    }
  };

  const run = async () => {
    if (!embedder || !corpus || !query.trim()) return;
    setBusy(true);
    try {
      const [vec] = await embedder([query.trim()]);
      const found: Hit[] = [];
      for (const key of KEYS) {
        const { dim, vectors } = corpus.vecs[key];
        for (const r of nearest(vec, vectors, dim, 3)) {
          found.push({ key, index: r.index, score: r.score });
        }
      }
      setHits(found);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <p className="note" style={{ padding: "16px 0 0" }}>
        文をひとつ打つと、<strong>あなたのブラウザの中で</strong>それを埋め込み、
        三つの版のどこに近いかを探します。文はどこにも送られません ——
        このサイトはサーバ関数を一つも持っていないので、送りようがありません。
      </p>

      {stage === "idle" && (
        <div className="verdict">
          <b>モデルは頼まれるまで取りに行きません。</b>{" "}
          {(MODEL_BYTES / 1024 / 1024).toFixed(0)} MB の量子化モデルと、
          焼いてある埋め込み 4.0 MB を読み込みます。回線の負担はあなたのものなので、
          押すまで一切通信しません。
          <div style={{ marginTop: 12 }}>
            <button type="button" className="chip" onClick={start}>
              モデルを読み込む({(MODEL_BYTES / 1024 / 1024).toFixed(0)} MB)
            </button>
          </div>
          <p className="note" style={{ marginTop: 10 }}>
            モデルは <code>{MODEL_ID}</code> を Hugging Face から直接取ります。
            公開面が量子化版を使うぶん、焼いてある fp32 とは cos 中央 0.995 ずれますが、
            探し当てる文は変わりません(独英日 各 40 件で 1 位的中 40/40)。
          </p>
        </div>
      )}

      {stage === "loading" && (
        <div className="verdict">
          <b>読み込んでいます。</b> {message}
          <div className="bar" aria-hidden="true">
            <span style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </div>
      )}

      {stage === "error" && (
        <div className="verdict">
          <b>読み込めませんでした。</b> {message}
        </div>
      )}

      {stage === "ready" && corpus && (
        <>
          <div className="controls">
            <input
              className="query"
              type="text"
              value={query}
              placeholder="独語でも英語でも日本語でも。例: ein ungeheueres Ungeziefer"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") run();
              }}
            />
            <button type="button" className="chip" onClick={run} disabled={busy}>
              {busy ? "探しています…" : "縫う"}
            </button>
          </div>

          {hits && (
            <div className="hits">
              {KEYS.map((key) => (
                <section key={key} className="cell" data-edition={key}>
                  <h3>{LABEL[key]}</h3>
                  {hits
                    .filter((h) => h.key === key)
                    .map((h) => (
                      <p key={`${key}-${h.index}`}>
                        <span className="score">{h.score.toFixed(3)}</span>
                        {corpus.texts[key].sentences[h.index]}
                      </p>
                    ))}
                </section>
              ))}
            </div>
          )}
        </>
      )}
    </>
  );
}
