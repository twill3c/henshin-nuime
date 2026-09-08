"""G-18 の期待値を作り直す道具。

`tests/fixtures/browser-embed.json` は**局所で量子化モデルを回した結果**である。
作った手順が版に残っていなければ、ブラウザと食い違ったときに
「ブラウザが違う」のか「期待値の作り方が違った」のかを切り分けられない。
実際 L10 でそこを踏んだ(SPEC §3.12)。

**一文ずつ回す。** ブラウザ(transformers.js)は照会を一文ずつ通す。
量子化モデルでは、まとめて通すか一文ずつかで出力が変わるので、
期待値もそう作らないと揃わない。ここが**揃える規則**の一つである(HC-073)。

  .venv/Scripts/python.exe scripts/make_g18_fixture.py            # 期待値だけ
  .venv/Scripts/python.exe scripts/make_g18_fixture.py --measure  # 実測も取り直す
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import embed as embed_mod  # noqa: E402

DATA_DIR = ROOT / "public" / "data"
FIXTURE = ROOT / "tests" / "fixtures" / "browser-embed.json"
QUANT_NAME = "model_quantized.onnx"
DIM = 384

# 照会に使う文。**版ごとに一つずつ**取り、三言語すべてに引き当たるかを見る。
# 番号は焼いた本文の通し番号(公開面が使うのと同じ番号)。
CASES = [("de", 401), ("en", 740), ("ja", 475)]

MEASURE_SAMPLE = 24   # fp32 対 量子化を見る文数
RANK1_SAMPLE = 40     # 版ごとに何件の照会で 1 位的中を数えるか
MEASURE_SEED = 20260908


def quantized_encoder() -> embed_mod.OnnxEncoder:
    path = embed_mod.MODEL_DIR / QUANT_NAME
    if not path.exists():
        raise SystemExit(f"{path} が無い。README の入手手順を見ること")
    return embed_mod.OnnxEncoder(path)


def embed_one_at_a_time(texts, enc, tok) -> np.ndarray:
    """**一文ずつ**通す。ブラウザと同じ回し方。まとめない。"""
    out = []
    for t in texts:
        ids, mask = embed_mod.encode_batch(tok, [t])
        out.append(embed_mod.mean_pool(enc.hidden_states(ids, mask), mask))
    return np.concatenate(out, axis=0).astype(np.float32)


def load_baked() -> dict[str, dict]:
    """焼いた本文と埋め込み(fp32)を、公開面が読むのと同じ形で読む。"""
    baked: dict[str, dict] = {}
    for key in ("de", "en", "ja"):
        text = json.loads((DATA_DIR / f"text-{key}.json").read_text(encoding="utf-8"))
        raw = (DATA_DIR / f"emb-{key}.bin").read_bytes()
        vecs = np.frombuffer(raw, dtype="<f4").reshape(-1, DIM)
        if len(vecs) != len(text["sentences"]):
            raise SystemExit(
                f"{key}: 埋め込み {len(vecs)} 本に対して本文 {len(text['sentences'])} 文"
            )
        baked[key] = {"sentences": text["sentences"], "vectors": vecs}
    return baked


def cosine(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    """公開面と同じ式。**L2 正規化はモデルに入っていない**ので、ここで割る。"""
    denom = np.linalg.norm(q) * np.linalg.norm(m, axis=1)
    denom[denom == 0] = 1.0
    return (m @ q) / denom


def nearest(q: np.ndarray, baked: dict) -> dict:
    hits = {}
    for key, v in baked.items():
        sims = cosine(q, v["vectors"])
        i = int(np.argmax(sims))
        hits[key] = {
            "index": i,
            "score": round(float(sims[i]), 4),
            "text": v["sentences"][i],
        }
    return hits


def measure(baked: dict, enc, tok) -> dict:
    """fp32 と量子化がどれだけ離れるかを、その場で測り直す。"""
    rng = np.random.default_rng(MEASURE_SEED)
    de = baked["de"]
    idx = rng.choice(len(de["sentences"]), MEASURE_SAMPLE, replace=False)
    texts = [de["sentences"][i] for i in idx]
    q = embed_one_at_a_time(texts, enc, tok)
    f = de["vectors"][idx]
    cos = np.sum(q * f, axis=1) / (
        np.linalg.norm(q, axis=1) * np.linalg.norm(f, axis=1)
    )
    hit = 0
    total = 0
    for key, v in baked.items():
        pick = rng.choice(len(v["sentences"]), RANK1_SAMPLE, replace=False)
        qs = embed_one_at_a_time([v["sentences"][i] for i in pick], enc, tok)
        for row, want in zip(qs, pick):
            total += 1
            hit += int(np.argmax(cosine(row, v["vectors"])) == want)
    return {
        "fp32_vs_quantized": {
            "max_abs_diff": round(float(np.max(np.abs(q - f))), 4),
            "cos_median": round(float(np.median(cos)), 6),
            "cos_min": round(float(np.min(cos)), 6),
            "sample": MEASURE_SAMPLE,
        },
        "quantized_query_rank1": {
            "hit": hit,
            "of": total,
            "note": f"量子化の照会 → fp32 で焼いた本文。独英日 各 {RANK1_SAMPLE} 件",
        },
    }


def batching_gap(baked: dict, enc, tok) -> dict:
    """**まとめて 対 一文ずつ**を、量子化と fp32 の両方で測る。

    L10 で期待値が落ちた原因そのもの。数字を作文しないよう、道具の中に置く。
    """
    texts = [baked[k]["sentences"][i] for k, i in CASES]
    ids, mask = embed_mod.encode_batch(tok, texts)
    out: dict[str, dict] = {}
    for name, e in (("quantized", enc), ("fp32", embed_mod.onnx_encoder())):
        batched = embed_mod.mean_pool(e.hidden_states(ids, mask), mask)
        single = embed_one_at_a_time(texts, e, tok)
        cos = np.sum(batched * single, axis=1) / (
            np.linalg.norm(batched, axis=1) * np.linalg.norm(single, axis=1)
        )
        out[name] = {
            "max_abs_diff": float(f"{np.max(np.abs(batched - single)):.3e}"),
            "cos_min": round(float(np.min(cos)), 6),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="fp32 対 量子化・1 位的中も取り直す(数分かかる)")
    args = ap.parse_args(argv)

    baked = load_baked()
    tok = embed_mod.load_tokenizer()
    enc = quantized_encoder()

    cases = []
    for key, index in CASES:
        text = baked[key]["sentences"][index]
        q = embed_one_at_a_time([text], enc, tok)[0]
        cases.append({"edition": key, "index": index, "text": text,
                      "expected": nearest(q, baked)})

    old = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else {}
    gap = batching_gap(baked, enc, tok)
    payload = {
        "note": ("局所で量子化モデルを**一文ずつ**回して出した期待値。"
                 "ブラウザ(transformers.js)が同じモデルファイルを同じ回し方で通して"
                 "同じ値を出すかを G-18 で確かめる。本文側は fp32 で焼いたもの。"
                 " 作り直し: scripts/make_g18_fixture.py"),
        "why_one_at_a_time": (
            "量子化モデルでは、まとめて通すか一文ずつ通すかで出力が変わる。"
            f"実測(この {len(CASES)} 文): "
            f"fp32 は最大差 {gap['fp32']['max_abs_diff']:.3e}、"
            f"量子化は最大差 {gap['quantized']['max_abs_diff']:.3e}・"
            f"cos 最小 {gap['quantized']['cos_min']}。"
            "ブラウザは照会を一文ずつ回すので、期待値もそう作らないと揃わない。"),
        "measured": old.get("measured", "2026-09-08"),
        "batching": gap,
        "fp32_vs_quantized": old.get("fp32_vs_quantized"),
        "quantized_query_rank1": old.get("quantized_query_rank1"),
        "cases": cases,
    }
    if args.measure:
        payload.update(measure(baked, enc, tok))
        payload["measured"] = __import__("datetime").date.today().isoformat()

    FIXTURE.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    print(f"{FIXTURE.relative_to(ROOT)} を書いた({len(cases)} 件)")
    print(f"  まとめて 対 一文ずつ: fp32 {gap['fp32']['max_abs_diff']:.3e} / "
          f"量子化 {gap['quantized']['max_abs_diff']:.3e}")


if __name__ == "__main__":
    main()
