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
    from . import align, ingest, sentences
else:  # スクリプトとして直接起動されたとき(HC-174)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import align, ingest, sentences


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
) -> Agreement:
    """独↔英でのみ意味を持つ。両版の段落構造が一致していることを前提に検算する。"""
    po_src = paragraph_ordinals(src)
    po_dst = paragraph_ordinals(dst)
    n_src, n_dst = max(po_src) + 1, max(po_dst) + 1
    if n_src != n_dst:
        raise ValueError(
            f"段落一致率は段落構造が一致する組でしか定義できない({n_src} 対 {n_dst})"
        )
    pairs = align.links(beads)
    matched = sum(1 for i, j in pairs if po_src[i] == po_dst[j])
    return Agreement(len(pairs), matched)


@dataclass(frozen=True)
class Refinement:
    paragraphs: int
    violations: int

    @property
    def rate(self) -> float:
        return self.violations / self.paragraphs if self.paragraphs else 0.0


def refinement_violations(
    beads: Sequence[align.Bead],
    src: Sequence[sentences.Sentence],
    dst: Sequence[sentences.Sentence],
) -> Refinement:
    """dst の一段落が src の複数段落にまたがった件数。dst が細かい側。"""
    po_src = paragraph_ordinals(src)
    po_dst = paragraph_ordinals(dst)
    spans: dict[int, set[int]] = {}
    for i, j in align.links(beads):
        spans.setdefault(po_dst[j], set()).add(po_src[i])
    violations = sum(1 for v in spans.values() if len(v) > 1)
    return Refinement(len(spans), violations)


PAIRS = [
    ("de_pg22367", "en_pg5200"),
    ("de_pg22367", "ja_aozora49866"),
    ("en_pg5200", "ja_aozora49866"),
]


def run() -> dict[tuple[str, str], dict[str, object]]:
    sents = sentences.load_all()
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
            row[name] = entry
        out[(a, b)] = row
    return out


def main() -> None:
    for (a, b), row in run().items():
        print(f"--- {a} → {b}  (c={row['c']:.4f}) ---")
        for name in ("gale_church", "diagonal"):
            e = row[name]
            agr = e["paragraph_agreement"]
            agr_s = "—(段落構造が違う)" if agr is None else f"{agr:.4f}"
            print(f"  {name:12s} 対応 {e['links']:5d}  段落一致率 {agr_s}"
                  f"  細分の破れ {e['refinement_violations']:4d}/{e['refinement_paragraphs']}")


if __name__ == "__main__":
    main()
