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


def bake_zure(sents, results) -> dict:
    """画面「ずれの図録」の材料(F-11)。

    **一つの数字にまとめない。** 分割・併合の形、段落一致率、細分の破れ、
    被覆、三角整合、そして訳者差 —— どれも別のものを見ているので、
    並べて出して読み手に比べさせる。良い手法の結果だけを出すと対照が働かない。
    """
    out: dict[str, object] = {"pairs": {}}
    for pair_key, (a, b) in PAIRS:
        row = results[(a, b)]
        methods: dict[str, object] = {}
        for m in METHODS:
            if m not in row:
                continue
            e = row[m]
            methods[m] = {
                "shapes": e["shapes"],
                "links": e["links"],
                "paragraph_agreement": e.get("paragraph_agreement"),
                "paragraph_agreement_common": e.get("paragraph_agreement_common"),
                "refinement_violations": e["refinement_violations"],
                "refinement_paragraphs": e["refinement_paragraphs"],
                "refinement_violations_common": e["refinement_violations_common"],
                "refinement_paragraphs_common": e["refinement_paragraphs_common"],
                "coverage_src": e["coverage_src"],
                "coverage_dst": e["coverage_dst"],
            }
        out["pairs"][pair_key] = {
            "c": row["c"],
            "n_src": row["n_src"],
            "n_dst": row["n_dst"],
            "common_src_sentences": row["common_src_sentences"],
            "common_dst_paragraphs": row["common_dst_paragraphs"],
            "methods": methods,
            # 置換検定の結果もそのまま焼く。**画面で数字を打ち直さない** ——
            # 同じ数を二箇所に書けば、必ず片方だけが古びる。
            "tests": {k: {"difference": r.observed, "p": r.p_display}
                      for k, r in row["tests"].items()},
        }

    tri = evaluate.triangle_verdict(results)
    out["triangle"] = {
        "n_src": tri["n_src"],
        # **片側だけの件数も出す。** 分母(compared)がどれだけ小さいかを
        # 見せないと、整合率だけが独り歩きする。
        "methods": {m: {"rate": tri[m].rate, "compared": tri[m].compared,
                        "agreed": tri[m].agreed,
                        "only_direct": tri[m].only_direct,
                        "only_composed": tri[m].only_composed,
                        "neither": tri[m].neither}
                    for m in tri["methods"]},
        "null": {m: sorted(v) for m, v in tri["null"].items()},
    }

    zs = evaluate.translator_lengths(results)
    if zs:
        out["translators"] = zs

    # 外挿検証(F-14 / G-09)。**英訳の本文は一字も配らない**(HC-255)ので、
    # ここに載るのは段落数などの集計値だけである。
    ex_dir = Path(__file__).resolve().parent.parent / "data" / "extrapolation"
    ex: dict[str, object] = {}
    for name in ("structure", "align"):
        path = ex_dir / f"{name}.json"
        if path.exists():
            ex[name] = json.loads(path.read_text(encoding="utf-8"))
    if len(ex) == 2:
        ex["work"] = "Der Prozess / 審判"
        ex["english_is_public_domain"] = False
        out["extrapolation"] = ex
    return out


def _own_translation_stats(sents) -> dict:
    """自前和訳の進み具合。**分子と分母で出す。**"""
    from . import translate as tr

    src = tr.load_source()
    ja = tr.load_translation()
    fill = tr.fill_rate(src, ja)
    return {
        "paragraphs": fill["paragraphs"],
        "by_chapter": fill["by_chapter"],
        "chars": fill["chars"],
        "terms": len(tr.load_glossary()),
    }


def bake_words(sents, results) -> list[dict]:
    """画面「一語の変身」の材料(F-10)。

    登録簿の語ごとに、**その語が最初に現れる独語の文**と、そこから
    **埋め込み由来の縫い目でたどった**英訳・原田訳の文、そして自前訳の段落を並べる。

    **縫い目は説明ではなく、ここでも被験体である。** 訳語を比べるために縫い目を
    使うので、縫い目が外れていれば並ぶ文も外れる。だから何本の対応をたどったかを
    そのまま出す —— 0 本なら「引けなかった」と表示する。
    """
    from . import translate as tr

    glossary = tr.load_glossary()
    if not glossary:
        return []
    own = {(r.chapter, r.index): r.text for r in tr.load_translation()}
    de = sents["de_pg22367"]
    forward = {}
    for pair_key, (a, b) in PAIRS:
        rows = align.links(results[(a, b)]["embedding"]["beads"])
        m: dict[int, list[int]] = {}
        for i, j in rows:
            m.setdefault(i, []).append(j)
        forward[pair_key] = m

    out: list[dict] = []
    for term, entry in glossary.items():
        pat = tr.term_pattern(term, entry)
        hit = next((i for i, s in enumerate(de) if pat.search(s.text)), None)
        if hit is None:
            raise BakeError(f"登録語 {term} が独語本文に無い")
        s = de[hit]
        row = {
            "term": term,
            "kind": entry.get("kind", "term"),
            # **`ja` という名前は使えない。** 相手の版の文を `en` / `ja` で持つので、
            # 訳語を `ja` に入れると文の配列に上書きされて黙って消える(実際に踏んだ)。
            "term_ja": entry["ja"],
            "why": entry.get("why", ""),
            "count": entry["count"],
            "first": entry["first"],
            "de": {"index": hit, "text": s.text},
        }
        for key, pair_key, eid in (("en", "de_en", "en_pg5200"),
                                   ("ja", "de_ja", "ja_aozora49866")):
            idx = forward[pair_key].get(hit, [])
            row[key] = [{"index": j, "text": sents[eid][j].text} for j in idx]
        text = own.get((s.chapter, s.paragraph))
        row["own"] = {"paragraph": f"{s.chapter}-{s.paragraph}", "text": text}
        if not isinstance(row.get("term_ja"), str) or not row["term_ja"]:
            raise BakeError(f"{term}: 訳語が文の配列に食われている")
        out.append(row)
    return out


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

    words = bake_words(sents, results)
    if words:
        files["words.json"] = {"terms": words}

    files["zure.json"] = bake_zure(sents, results)

    files["manifest.json"] = {
        "has_attention": attention is not None,
        "chapters": sorted({s.chapter for s in sents[EDITIONS[0][1]]}),
        "methods": list(METHODS),
        "pairs": [k for k, _ in PAIRS],
        # 章をまたぐ対応の件数は**手法ごとに**出す。一つにまとめない。
        "cross_chapter_links": cross,
        # 自前和訳の充填率。**分子と分母で出す** —— 「何割」だけを書かない。
        "own_translation": _own_translation_stats(sents),
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


def bake_embeddings() -> dict[str, int]:
    """焼いた埋め込みを float32 の生バイトで出す。

    **JSON にしない。** 2,694 文 × 384 次元を文字にすると 20 MB を超えるが、
    生の float32 なら 4.1 MB で済む。読む側は `Float32Array` に載せるだけでよい。

    この 4.1 MB は**モデル(118 MB)を頼まれたときにだけ**取りに行くので、
    既定の転送量には入らない(N-03)。
    """
    import numpy as np

    from . import embed as embed_mod  # torch を引かないよう遅らせて読む

    try:
        vecs = embed_mod.load_cached()
    except embed_mod.ModelMissing:
        return {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for key, eid in EDITIONS:
        v = np.asarray(vecs[eid], dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != 384:
            raise BakeError(f"{eid} の埋め込みの形が {v.shape}")
        path = OUT_DIR / f"emb-{key}.bin"
        path.write_bytes(v.tobytes(order="C"))
        sizes[path.name] = path.stat().st_size
    return sizes


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
    for name, size in sorted(bake_embeddings().items()):
        total += size
        print(f"  {name}: {size/1024:.0f} KB(頼まれたときだけ配る)")
    print(f"合計 {total/1024:.0f} KB → {OUT_DIR}")


if __name__ == "__main__":
    main()
