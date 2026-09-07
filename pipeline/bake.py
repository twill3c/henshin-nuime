"""公開面が読むデータを焼く。**ここから先はビルド時に固まる。**

公開面はサーバ関数を持たない(N-01)。だから縫い目も本文も、この工程で
`public/data/` に焼き込み、ブラウザは静的ファイルを読むだけにする。

**文の番号は版ごとの通し番号(大域)で書く。** 最初は章ごとに自己完結する
ファイルに分けようとしたが、**章をまたぐ対応がある**ので落ちる対応が出る。
実測(2026-09-07・独→英 / 独→日):

    埋め込み      0 本 / 0 本
    長さモデル    1 本 / 48 本
    対角線       15 本 / 56 本

埋め込みだけが章を一本もまたがない。章はアライナに一度も渡していない(G-03)ので、
これは held-out の観察である —— ただし**埋め込みの性質であって、手法一般の性質ではない**。
一手法だけを測って全手法の話に広げかけたのを、焼く前のガードが止めた。

**三手法とも焼く。** この企画は解剖台なので、「埋め込みだと繋がるが対角線だと
繋がらない」を読み手が切り替えて見られることが本体である。良い手法の結果だけを
出すと、対照が対照として働かない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__:
    from . import align, evaluate, rights, sentences
else:  # スクリプトとして直接起動されたとき(HC-174)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import align, evaluate, rights, sentences

OUT_DIR = Path(__file__).resolve().parent.parent / "public" / "data"

# 公開面に出す版の並びと短い名前。順序は独→英→日で固定する。
EDITIONS = [
    ("de", "de_pg22367"),
    ("en", "en_pg5200"),
    ("ja", "ja_aozora49866"),
]
PAIRS = [("de_en", ("de_pg22367", "en_pg5200")),
         ("de_ja", ("de_pg22367", "ja_aozora49866"))]
METHODS = ("embedding", "gale_church", "diagonal")


class BakeError(RuntimeError):
    """焼く前提が崩れたときに投げる。黙って壊れた画面を出さない。"""


def chapter_offsets(sents) -> list[int]:
    """章の開始位置(通し番号)。文が章の順に並んでいることを検算する。"""
    offsets: list[int] = []
    current = 0
    for i, s in enumerate(sents):
        if s.chapter != current:
            if s.chapter != current + 1:
                raise BakeError(f"章が飛んでいる: {current} → {s.chapter}(文 {i})")
            offsets.append(i)
            current = s.chapter
    if not offsets:
        raise BakeError("章の切れ目が見つからない")
    return offsets


def build() -> dict[str, object]:
    sents = sentences.load_all()
    results = evaluate.run()
    ledger = rights.load_ledger()

    files: dict[str, object] = {}
    for key, eid in EDITIONS:
        v = sents[eid]
        files[f"text-{key}.json"] = {
            "id": eid,
            "sentences": [s.text for s in v],
            "chapter": [s.chapter for s in v],
            "paragraph": _global_paragraphs(v),
        }

    links: dict[str, object] = {}
    cross: dict[str, dict[str, int]] = {}
    for pair_key, (a, b) in PAIRS:
        per_method: dict[str, list[list[int]]] = {}
        per_cross: dict[str, int] = {}
        for m in METHODS:
            if m not in results[(a, b)]:
                continue
            pairs = align.links(results[(a, b)][m]["beads"])
            per_method[m] = [[i, j] for i, j in pairs]
            per_cross[m] = sum(1 for i, j in pairs
                               if sents[a][i].chapter != sents[b][j].chapter)
        if not per_method:
            raise BakeError(f"{pair_key} の対応が一つも無い")
        links[pair_key] = per_method
        cross[pair_key] = per_cross
    files["links.json"] = links

    attention = _bake_attention(sents)
    if attention is not None:
        files["attention.json"] = attention

    files["manifest.json"] = {
        "has_attention": attention is not None,
        "chapters": sorted({s.chapter for s in sents[EDITIONS[0][1]]}),
        "methods": list(METHODS),
        "pairs": [k for k, _ in PAIRS],
        # 章をまたぐ対応の件数は**手法ごとに**出す。一つにまとめない。
        "cross_chapter_links": cross,
        "editions": [
            {
                "key": key,
                "id": eid,
                "lang": rights.get(ledger, eid)["lang"],
                "title": rights.get(ledger, eid)["title"],
                "translator": rights.get(ledger, eid)["translator"],
                "source_url": rights.get(ledger, eid)["source_url"],
                "attribution": rights.get(ledger, eid)["attribution"],
                "sentences": len(sents[eid]),
                "chapter_offsets": chapter_offsets(sents[eid]),
            }
            for key, eid in EDITIONS
        ],
    }
    return files


def _bake_attention(sents) -> dict[str, object] | None:
    """cross-attention の点数表と、目玉の判定を焼く。

    点数表は帯の中にしか値が無い(全体の 1%)ので**疎な三つ組**で持つ。
    密に書くと 696×771 で 2 MB を超えるが、三つ組なら 82 KB で済む。

    **落ちた判定もそのまま焼く。** 図に出すのは結果であって、良い結果ではない。
    """
    import numpy as np

    if not evaluate.ATTENTION_SCORES.exists():
        return None
    scores = np.load(evaluate.ATTENTION_SCORES)
    verdict = evaluate.attention_verdict()
    nz = np.argwhere(scores > 0)
    a, b = evaluate.ATTENTION_PAIR
    return {
        "pair": "de_en",
        "shape": list(scores.shape),
        # [独語の文番号, 英語の文番号, 注意のシェア]。小数 3 桁で足りる(値域 0〜1)
        "cells": [[int(i), int(j), round(float(scores[i, j]), 3)] for i, j in nz],
        "chance_share": verdict["chance_share"],
        "links": [[i, j] for i, j in align.links(verdict["attention"]["beads"])],
        "verdict": {
            "attention": {
                "links": verdict["attention"]["links"],
                "paragraph_agreement": verdict["attention"]["paragraph_agreement"],
                "coverage_src": verdict["attention"]["coverage_src"],
            },
            "diagonal": {
                "links": verdict["diagonal"]["links"],
                "paragraph_agreement": verdict["diagonal"]["paragraph_agreement"],
            },
            "difference": verdict["test"].observed,
            "p_value": verdict["test"].p_value,
            "p_display": verdict["test"].p_display,
            "passed": bool(verdict["test"].observed > 0
                           and verdict["test"].p_value < 0.01),
            "threshold": 0.01,
        },
    }


def _global_paragraphs(sents) -> list[int]:
    """章をまたいで 0 から通した段落番号。表示の区切りに使う。"""
    out: list[int] = []
    seen: dict[tuple[int, int], int] = {}
    for s in sents:
        key = (s.chapter, s.paragraph)
        if key not in seen:
            seen[key] = len(seen)
        out.append(seen[key])
    return out


def main() -> None:
    files = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, payload in sorted(files.items()):
        path = OUT_DIR / name
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        path.write_text(text, encoding="utf-8")
        total += len(text.encode("utf-8"))
        print(f"  {name}: {len(text.encode('utf-8'))/1024:.0f} KB")
    print(f"合計 {total/1024:.0f} KB → {OUT_DIR}")


if __name__ == "__main__":
    main()
