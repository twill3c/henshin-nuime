"""縫い目を段落オラクルで測る。

**段落情報はここで初めて登場する。** アライナ(`pipeline/align.py`)には
文の本文しか渡っていないので、この評価は held-out である(G-03)。

二つの物差しがある。使える場所が違う。

**段落一致率**(独↔英でのみ定義できる)
  L0 の実測で、独と英は段落構造が完全に一致する(97 段落・章別 [30,29,38])。
  だから「独の第 k 段落の文が、英の第 k 段落の文に結ばれたか」を直接数えられる。
  これは正解ラベルであって、推定ではない。

**細分の破れ**(独↔日で使う)
  日本語の段落は独語より細かい(165 対 97)。もし日本語が独語の段落の**細分**なら、
  日本語の一段落は独語の一段落の中に収まるはずである。収まらなかった数を数える。
  **ただしこの数は二つのものを混ぜて測っている** —— アライナの誤りと、
  「細分である」という仮説そのものの誤り。だから対照(対角線)との差だけを読む。
  この数字を単独で「細分性が成り立った/成り立たない」の根拠にしてはならない。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__:
    from . import align, embed, ingest, sentences, stats
else:  # スクリプトとして直接起動されたとき(HC-174)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import align, embed, ingest, sentences, stats


def paragraph_ordinals(sents: Sequence[sentences.Sentence]) -> list[int]:
    """文ごとの通し段落番号(章をまたいで 0 から数える)。"""
    ordinals: list[int] = []
    seen: dict[tuple[int, int], int] = {}
    for s in sents:
        key = (s.chapter, s.paragraph)
        if key not in seen:
            seen[key] = len(seen)
        ordinals.append(seen[key])
    return ordinals


@dataclass(frozen=True)
class Agreement:
    links: int
    matched: int

    @property
    def rate(self) -> float:
        return self.matched / self.links if self.links else 0.0


def paragraph_agreement(
    beads: Sequence[align.Bead],
    src: Sequence[sentences.Sentence],
    dst: Sequence[sentences.Sentence],
    *,
    src_subset: set[int] | None = None,
) -> Agreement:
    """独↔英でのみ意味を持つ。両版の段落構造が一致していることを前提に検算する。

    `src_subset` を渡すと、その文から出た対応だけで測る。手法ごとに被覆率が違うと
    分母が変わり、**飛ばすほど有利になる**ので、比べるときは共通部分で測り直す。
    """
    po_src = paragraph_ordinals(src)
    po_dst = paragraph_ordinals(dst)
    n_src, n_dst = max(po_src) + 1, max(po_dst) + 1
    if n_src != n_dst:
        raise ValueError(
            f"段落一致率は段落構造が一致する組でしか定義できない({n_src} 対 {n_dst})"
        )
    pairs = align.links(beads)
    if src_subset is not None:
        pairs = [(i, j) for i, j in pairs if i in src_subset]
    matched = sum(1 for i, j in pairs if po_src[i] == po_dst[j])
    return Agreement(len(pairs), matched)


@dataclass(frozen=True)
class Refinement:
    paragraphs: int
    violations: int

    @property
    def rate(self) -> float:
        return self.violations / self.paragraphs if self.paragraphs else 0.0


def covered_dst_paragraphs(
    beads: Sequence[align.Bead], dst: Sequence[sentences.Sentence]
) -> set[int]:
    po_dst = paragraph_ordinals(dst)
    return {po_dst[j] for _, j in align.links(beads)}


def refinement_violations(
    beads: Sequence[align.Bead],
    src: Sequence[sentences.Sentence],
    dst: Sequence[sentences.Sentence],
    *,
    dst_subset: set[int] | None = None,
) -> Refinement:
    """dst の一段落が src の複数段落にまたがった件数。dst が細かい側。

    `dst_subset` を渡すと、その段落だけで測る。**分母は手法ごとに違う** ——
    相手の付かなかった段落は数に入らないので、飛ばすほど破れが減って見える。
    比べるときは全手法が覆った段落だけで測り直す。
    """
    po_src = paragraph_ordinals(src)
    po_dst = paragraph_ordinals(dst)
    spans: dict[int, set[int]] = {}
    for i, j in align.links(beads):
        p = po_dst[j]
        if dst_subset is not None and p not in dst_subset:
            continue
        spans.setdefault(p, set()).add(po_src[i])
    violations = sum(1 for v in spans.values() if len(v) > 1)
    return Refinement(len(spans), violations)


PAIRS = [
    ("de_pg22367", "en_pg5200"),
    ("de_pg22367", "ja_aozora49866"),
    ("en_pg5200", "ja_aozora49866"),
]


def run(*, with_embeddings: bool = True) -> dict[tuple[str, str], dict[str, object]]:
    """三手法を同じ段落オラクルで測る。

    埋め込みが手元に無ければ、その手法だけ飛ばして残り二つを出す
    (黙って全部を諦めない)。
    """
    sents = sentences.load_all()
    vecs: dict[str, object] | None = None
    if with_embeddings:
        try:
            vecs = embed.load_cached()
        except embed.ModelMissing as e:
            print(f"[埋め込みなしで続行] {e}")
            vecs = None

    out: dict[tuple[str, str], dict[str, object]] = {}
    for a, b in PAIRS:
        src_t = [s.text for s in sents[a]]
        dst_t = [s.text for s in sents[b]]
        c = align.stretch_from_totals(src_t, dst_t)
        methods = {
            "gale_church": align.align(src_t, dst_t, c=c),
            "diagonal": align.align_diagonal(len(src_t), len(dst_t)),
        }
        row: dict[str, object] = {"c": c, "n_src": len(src_t), "n_dst": len(dst_t)}
        if vecs is not None:
            chance = align.chance_similarity(vecs[a], vecs[b])
            row["chance"] = chance
            methods["embedding"] = align.align_embeddings(
                vecs[a], vecs[b], baseline=chance)
        for name, beads in methods.items():
            entry: dict[str, object] = {
                # 対応そのものを持たせる。DP は 3 組で 40 秒かかるので、
                # 検査が同じものを引き直さずに済むようにする。
                "beads": beads,
                "shapes": {f"{k[0]}-{k[1]}": v
                           for k, v in sorted(align.shape_counts(beads).items())}
            }
            try:
                agr = paragraph_agreement(beads, sents[a], sents[b])
                entry["paragraph_agreement"] = agr.rate
                entry["links"] = agr.links
            except ValueError:
                entry["paragraph_agreement"] = None
                entry["links"] = len(align.links(beads))
            ref = refinement_violations(beads, sents[a], sents[b])
            entry["refinement_violations"] = ref.violations
            entry["refinement_paragraphs"] = ref.paragraphs
            cov_src, cov_dst = align.coverage(beads, len(src_t), len(dst_t))
            entry["coverage_src"] = cov_src
            entry["coverage_dst"] = cov_dst
            row[name] = entry

        # 手法ごとに被覆が違うので、**全手法が覆った部分だけ**でもう一度測る。
        # 飛ばすほど有利になる物差しを、そのまま並べて比べてはいけない。
        names = [n for n in METHODS if n in row]
        common_src = set.intersection(*(
            {i for bd in row[n]["beads"] for i in range(bd.i0, bd.i1)
             if bd.shape[1] > 0} for n in names))
        common_dst_p = set.intersection(*(
            covered_dst_paragraphs(row[n]["beads"], sents[b]) for n in names))
        row["common_src_sentences"] = len(common_src)
        row["common_dst_paragraphs"] = len(common_dst_p)
        for n in names:
            beads = row[n]["beads"]
            try:
                agr = paragraph_agreement(beads, sents[a], sents[b],
                                          src_subset=common_src)
                row[n]["paragraph_agreement_common"] = agr.rate
                row[n]["links_common"] = agr.links
            except ValueError:
                row[n]["paragraph_agreement_common"] = None
                row[n]["links_common"] = None
            ref = refinement_violations(beads, sents[a], sents[b],
                                        dst_subset=common_dst_p)
            row[n]["refinement_violations_common"] = ref.violations
            row[n]["refinement_paragraphs_common"] = ref.paragraphs

        row["tests"] = _significance(row, names, sents[a], sents[b],
                                     common_src, common_dst_p)
        out[(a, b)] = row
    return out


def _significance(row: dict, names: list[str],
                  src: Sequence[sentences.Sentence],
                  dst: Sequence[sentences.Sentence],
                  common_src: set[int],
                  common_dst_p: set[int]) -> dict[str, object]:
    """埋め込みと他の手法との差を、対応のある置換検定に掛ける。

    **単位も並べ替えのブロックも、結果を見てから選んでいない。**
    段落一致率は「文を単位・段落をブロック」— どちらも本文の構造から決まる。
    細分の破れは「段落を単位」で、ブロックの大きさだけは構造から決まらないので
    複数の値を並べる(`REFINEMENT_BLOCK_SIZES`)。
    """
    if "embedding" not in names:
        return {}
    tests: dict[str, object] = {}
    po_src = paragraph_ordinals(src)

    # 段落一致率(独↔英でのみ定義できる)
    if row["embedding"]["paragraph_agreement"] is not None:
        units = sorted(common_src)
        blocks = [po_src[i] for i in units]
        base = sentence_agreement_outcomes(row["embedding"]["beads"], src, dst,
                                           common_src)
        for other in [n for n in names if n != "embedding"]:
            comp = sentence_agreement_outcomes(row[other]["beads"], src, dst,
                                               common_src)
            r = stats.paired_permutation([base[i] for i in units],
                                         [comp[i] for i in units], blocks)
            tests[f"agreement_vs_{other}"] = r

    # 細分の破れ(全組)
    units_p = sorted(common_dst_p)
    base_r = refinement_outcomes(row["embedding"]["beads"], src, dst, common_dst_p)
    for other in [n for n in names if n != "embedding"]:
        comp_r = refinement_outcomes(row[other]["beads"], src, dst, common_dst_p)
        for size in REFINEMENT_BLOCK_SIZES:
            blocks = [k // size for k in range(len(units_p))]
            r = stats.paired_permutation([base_r[p] for p in units_p],
                                         [comp_r[p] for p in units_p], blocks)
            tests[f"refinement_vs_{other}_block{size}"] = r
    return tests


METHODS = ("gale_church", "diagonal", "embedding")

# 細分の破れを検定するときの並べ替えブロックの大きさ。**一つに決めない。**
# 隣り合う段落は同じ誤りに巻き込まれるので、1 段落ずつ入れ替えると相関を壊して
# p 値が甘くなる。かといってブロックの大きさは構造から決まらないので、
# 複数の値で出して結果が幅に対して頑健かを見る。
REFINEMENT_BLOCK_SIZES = (1, 5, 10)


def sentence_agreement_outcomes(
    beads: Sequence[align.Bead],
    src: Sequence[sentences.Sentence],
    dst: Sequence[sentences.Sentence],
    subset: set[int],
) -> dict[int, float]:
    """src 文ごとに「その文の対応が全部正しい段落に落ちたか」を 1/0 で返す。

    対応そのものを単位にすると、同じ文から出た複数の対応が独立でないまま
    数に入る。文を単位にして、その中は「全部正しいか」に畳む。
    """
    po_src = paragraph_ordinals(src)
    po_dst = paragraph_ordinals(dst)
    hits: dict[int, list[bool]] = {i: [] for i in subset}
    for i, j in align.links(beads):
        if i in hits:
            hits[i].append(po_src[i] == po_dst[j])
    return {i: (1.0 if v and all(v) else 0.0) for i, v in hits.items()}


def refinement_outcomes(
    beads: Sequence[align.Bead],
    src: Sequence[sentences.Sentence],
    dst: Sequence[sentences.Sentence],
    subset: set[int],
) -> dict[int, float]:
    """dst 段落ごとに「破れていないか」を 1/0 で返す。"""
    po_src = paragraph_ordinals(src)
    po_dst = paragraph_ordinals(dst)
    spans: dict[int, set[int]] = {p: set() for p in subset}
    for i, j in align.links(beads):
        p = po_dst[j]
        if p in spans:
            spans[p].add(po_src[i])
    return {p: (1.0 if len(v) <= 1 else 0.0) for p, v in spans.items()}


ATTENTION_PAIR = ("de_pg22367", "en_pg5200")
ATTENTION_SCORES = (Path(__file__).resolve().parent.parent
                    / "data" / "nmt" / "attention_de_en.npy")


def attention_verdict(*, chance_share: float | None = None) -> dict[str, object]:
    """目玉(G-08)の判定。**attention 由来の縫い目を対角線と比べる。**

    比べるのは段落一致率で、単位は src 文・並べ替えのブロックはその文が属する段落
    (§3.5 と同じ取り方)。**attention は窓が許す帯の中でしか対応を組めない**ので、
    自由度は対角線より小さい。だから勝てば、勝ったぶんは本文から来ている。

    判定に使う「偶然の水準」は窓の構造から決まる値(既定は `nmt.CHANCE_SHARE`)で、
    結果を見てから選ぶつまみではない。
    """
    import json

    import numpy as np

    if not ATTENTION_SCORES.exists():
        raise FileNotFoundError(
            f"{ATTENTION_SCORES} が無い。`python -m pipeline.nmt` で作ること")
    if chance_share is None:
        # 学習時に記録した値を読む。**評価器から torch を引っぱらないため**に
        # `nmt` を import せず、成果物の側から取る。
        meta = json.loads((ATTENTION_SCORES.parent / "history.json")
                          .read_text(encoding="utf-8"))
        chance_share = float(meta["chance_share"])
    scores = np.load(ATTENTION_SCORES)
    sents = sentences.load_all()
    a, b = ATTENTION_PAIR
    n, m = scores.shape

    beads = {
        "attention": align.align_attention(scores, baseline=chance_share),
        "diagonal": align.align_diagonal(n, m),
    }
    out: dict[str, object] = {"n_src": n, "n_dst": m, "chance_share": chance_share}
    common = set.intersection(*(
        {i for bd in v for i in range(bd.i0, bd.i1) if bd.shape[1] > 0}
        for v in beads.values()))
    out["common_src_sentences"] = len(common)
    for name, bd in beads.items():
        agr = paragraph_agreement(bd, sents[a], sents[b])
        agr_c = paragraph_agreement(bd, sents[a], sents[b], src_subset=common)
        cov = align.coverage(bd, n, m)
        out[name] = {
            "beads": bd, "links": agr.links, "paragraph_agreement": agr.rate,
            "paragraph_agreement_common": agr_c.rate, "links_common": agr_c.links,
            "coverage_src": cov[0], "coverage_dst": cov[1],
        }

    po_src = paragraph_ordinals(sents[a])
    units = sorted(common)
    blocks = [po_src[i] for i in units]
    base = sentence_agreement_outcomes(beads["attention"], sents[a], sents[b], common)
    comp = sentence_agreement_outcomes(beads["diagonal"], sents[a], sents[b], common)
    out["test"] = stats.paired_permutation([base[i] for i in units],
                                           [comp[i] for i in units], blocks)
    return out


def main() -> None:
    for (a, b), row in run().items():
        chance = row.get("chance")
        head = f"--- {a} → {b}  (c={row['c']:.4f}"
        head += f", 偶然の水準={chance:.4f})" if chance is not None else ")"
        print(head + " ---")
        for name in METHODS:
            e = row.get(name)
            if e is None:
                continue
            agr = e["paragraph_agreement"]
            agr_s = "     —" if agr is None else f"{agr:.4f}"
            agrc = e["paragraph_agreement_common"]
            agrc_s = "     —" if agrc is None else f"{agrc:.4f}"
            print(f"  {name:12s} 対応 {e['links']:5d}  段落一致率 {agr_s}"
                  f"  細分の破れ {e['refinement_violations']:4d}/{e['refinement_paragraphs']}"
                  f"  被覆 {e['coverage_src']:.3f}/{e['coverage_dst']:.3f}")
            print(f"  {'':12s} 共通部分: 段落一致率 {agrc_s}"
                  f"  細分の破れ {e['refinement_violations_common']:4d}"
                  f"/{e['refinement_paragraphs_common']}")
        print(f"    (共通部分 = 全手法が対応をつけた src 文 "
              f"{row['common_src_sentences']} 件 / dst 段落 "
              f"{row['common_dst_paragraphs']} 件)")
        for name, r in row.get("tests", {}).items():
            print(f"    置換検定 {name:34s} 差 {r.observed:+.4f}  "
                  f"p {r.p_display}  ブロック {r.blocks}")


if __name__ == "__main__":
    main()
