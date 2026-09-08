// ブラウザ内推論。**三本目の経路**(G-18)。
//
// 経路A は onnxruntime(Python)、経路B は自前の NumPy 順伝播、
// そしてここが経路C —— transformers.js が同じ ONNX を WASM で回す。
//
// **照合は同じモデルファイルどうしで行う。** 公開面は量子化版(118 MB)を使うので、
// 焼いてある fp32 の埋め込みと直接は比べられない。実測(2026-09-08)で
// fp32 対 量子化は cos 中央 0.995136・最小 0.987924 ずれる。
// だから G-18 は「ブラウザの量子化 対 局所の量子化」で見る。
//
// **その食い違いは検索には効かない。** 量子化で埋め込んだ照会を fp32 で焼いた本文に
// 当てると、120 件すべてで正解が 1 位に来た(独英日 各 40 件・2026-09-08)。
// ずれ(cos 0.99)は、別の文との差(cos 0.26 が偶然の水準)よりはるかに小さい。
//
// **モデルは押しつけない。** 118 MB は利用者の回線の話なので、
// 頼まれるまで一切取りに行かない(N-03)。

export const MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
export const MODEL_BYTES = 118_308_126;
/** 本家 `sentence_bert_config.json` の max_seq_length。焼いた側と揃える。 */
export const MAX_SEQ_LEN = 128;

export type Embedder = (texts: string[]) => Promise<Float32Array[]>;

let cached: Promise<Embedder> | null = null;

/**
 * 埋め込み器を用意する。**呼ばれるまで何も取りに行かない。**
 * 二度目以降は同じものを返す(モデルを二重に落とさない)。
 */
export function loadEmbedder(
  onProgress?: (fraction: number, label: string) => void,
): Promise<Embedder> {
  if (cached) return cached;
  cached = (async () => {
    const { AutoTokenizer, AutoModel, env } = await import(
      "@huggingface/transformers"
    );
    // 手元のファイルは持たない。取りに行く先は一つだけにする。
    env.allowLocalModels = false;

    const progress = (info: { status?: string; progress?: number; file?: string }) => {
      if (!onProgress) return;
      if (info.status === "progress" && typeof info.progress === "number") {
        onProgress(info.progress / 100, info.file ?? "");
      } else if (info.status === "done") {
        onProgress(1, info.file ?? "");
      }
    };

    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: progress,
    });
    const model = await AutoModel.from_pretrained(MODEL_ID, {
      dtype: "q8", // 焼いた側と同じ量子化ファイル
      progress_callback: progress,
    });

    return async (texts: string[]) => {
      const inputs = tokenizer(texts, {
        padding: true,
        truncation: true,
        max_length: MAX_SEQ_LEN,
      });
      const out = await model(inputs);
      const hidden = out.last_hidden_state;
      const [b, t, d] = hidden.dims as [number, number, number];
      const h = hidden.data as Float32Array;
      const mask = inputs.attention_mask.data as BigInt64Array | Int32Array;

      // 平均プーリング。**L2 正規化はしない** —— 本家 modules.json に Normalize が無く、
      // 焼いた側もしていない。片方だけ正規化すると照合が黙って合わなくなる。
      const result: Float32Array[] = [];
      for (let n = 0; n < b; n += 1) {
        const vec = new Float32Array(d);
        let count = 0;
        for (let k = 0; k < t; k += 1) {
          const m = Number(mask[n * t + k]);
          if (!m) continue;
          count += 1;
          const base = (n * t + k) * d;
          for (let c = 0; c < d; c += 1) vec[c] += h[base + c];
        }
        const denom = Math.max(count, 1);
        for (let c = 0; c < d; c += 1) vec[c] /= denom;
        result.push(vec);
      }
      return result;
    };
  })();
  return cached;
}

/** 余弦。**どちらも正規化していない前提**で毎回割る。 */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

/** 焼いた埋め込み(float32 の生バイト)を読む。 */
export async function fetchEmbeddings(
  key: string,
): Promise<{ dim: number; vectors: Float32Array }> {
  const res = await fetch(`/data/emb-${key}.bin`);
  if (!res.ok) throw new Error(`emb-${key}.bin が読めない (${res.status})`);
  const buf = await res.arrayBuffer();
  return { dim: 384, vectors: new Float32Array(buf) };
}

/** `vectors` の中で `query` に最も近い上位 `k` 件。 */
export function nearest(
  query: ArrayLike<number>,
  vectors: Float32Array,
  dim: number,
  k: number,
): { index: number; score: number }[] {
  const n = vectors.length / dim;
  if (!Number.isInteger(n)) throw new Error("次元が合わない");
  const out: { index: number; score: number }[] = [];
  for (let i = 0; i < n; i += 1) {
    out.push({ index: i, score: cosine(query, vectors.subarray(i * dim, (i + 1) * dim)) });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, k);
}
