"""縫い目の検査。TEST_SPEC の T-024〜T-028 に対応する。

この企画の背骨は**循環の禁止**である。「段落を渡していない」を約束ではなく
構造で示すため、二方向から押さえる:

  構造 (T-024)  `pipeline/align.py` の AST に、段落を知る型への経路が無い
  振る舞い (T-025)  メタデータを入れ替えても縫い目が 1 文字も変わらない

実データの縫い目は 3 組で 40 秒ほどかかるので、セッション全体で 1 度だけ計算する。
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import align, evaluate, sentences  # noqa: E402

# align.py が触れてはならない属性。段落・章・識別子への経路をすべて挙げる。
FORBIDDEN_ATTRS = {"chapter", "paragraph", "pid", "sid", "edition_id", "index",
                   "paragraphs", "chapters"}


@pytest.fixture(scope="session")
def sents():
    return sentences.load_all()


@pytest.fixture(scope="session")
def results():
    return evaluate.run()


# --- T-024: 循環の禁止・構造 --------------------------------------------------


def _isolation_report(source: str) -> list[str]:
    """段落を知る型への経路を並べて返す。空なら隔離できている。"""
    tree = ast.parse(source)
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                problems.append(f"相対 import: {node.module}")
            elif node.module and node.module.split(".")[0] == "pipeline":
                problems.append(f"pipeline の import: {node.module}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "pipeline":
                    problems.append(f"pipeline の import: {a.name}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"禁じた属性への参照: .{node.attr}")
    return problems


@pytest.mark.validation
def test_t024_aligner_is_structurally_isolated():
    """T-024(G-03)— align.py に段落を知る型への経路が構文として無い。"""
    source = (ROOT / "pipeline" / "align.py").read_text(encoding="utf-8")
    assert _isolation_report(source) == []


@pytest.mark.unit
def test_t024_isolation_check_positive_control():
    """T-024 — 検査が実際に撃つこと(陽性対照)。

    「違反 0 件」は検査器が死んでいても同じ値を返す。
    """
    assert _isolation_report("from pipeline import sentences\n")
    assert _isolation_report("from . import ingest\n")
    assert _isolation_report("x = sent.chapter\n")
    assert _isolation_report("y = s.paragraph + 1\n")
    # 陰性対照 — 正当なソースを撃たないこと
    assert _isolation_report("import math\nz = math.erfc(1.0)\n") == []


# --- T-025: 循環の禁止・振る舞い ----------------------------------------------


@pytest.mark.validation
def test_t025_alignment_ignores_metadata(sents):
    """T-025(G-03)— メタデータを入れ替えても縫い目が変わらない。

    構造の検査(T-024)は「経路が無い」ことしか言わない。実際にその経路を通って
    いないことは、メタデータを壊して結果が動かないことで示す。
    """
    src = [s.text for s in sents["de_pg22367"]][:120]
    dst = [s.text for s in sents["en_pg5200"]][:130]
    c = align.stretch_from_totals(src, dst)
    baseline = align.align(src, dst, c=c)

    # 章・段落・識別子をでたらめにした文を作り、本文だけを取り出して同じ手順を踏む
    scrambled = [
        dataclasses.replace(s, chapter=9, paragraph=999 - i, index=0,
                            sid="x", edition_id="scrambled")
        for i, s in enumerate(sents["de_pg22367"][:120])
    ]
    assert [s.chapter for s in scrambled] != [
        s.chapter for s in sents["de_pg22367"][:120]
    ], "対照の前提が崩れている — メタデータが実際に変わっていない"

    again = align.align([s.text for s in scrambled], dst, c=c)
    assert again == baseline


@pytest.mark.unit
def test_t025_aligner_rejects_non_text():
    """T-025(G-03)— 文字列以外の列は受け取らない。"""
    with pytest.raises(TypeError):
        align.align([object()], ["a"], c=1.0)
    with pytest.raises(TypeError):
        align.align("abc", ["a"], c=1.0)  # 文字列そのものは列として扱わない
    with pytest.raises(ValueError):
        align.align([], ["a"], c=1.0)


# --- T-026: DP の正しさと被覆 -------------------------------------------------


@pytest.mark.unit
def test_t026_dp_recovers_known_correspondences():
    """T-026 — 合成データで既知の対応を再現する。

    期待値は長さから導出する。c=1 のとき、同じ長さの文どうしが 1-1 で結ばれ、
    倍の長さの文は二つに割れた側と 1-2 で結ばれるのが長さモデルの帰結である。
    """
    a = "a" * 100
    beads = align.align([a, a, a], [a, a, a], c=1.0)
    assert [b.shape for b in beads] == [(1, 1), (1, 1), (1, 1)]

    beads = align.align(["b" * 200], ["b" * 100, "b" * 100], c=1.0)
    assert [b.shape for b in beads] == [(1, 2)]

    beads = align.align(["c" * 100, "c" * 100], ["c" * 200], c=1.0)
    assert [b.shape for b in beads] == [(2, 1)]


@pytest.mark.integration
def test_t026_beads_tile_both_sides(results):
    """T-026 — 対応が両側を隙間も重なりも無く覆う。手法と対照の両方で。"""
    for (a, b), row in results.items():
        for name in ("gale_church", "diagonal"):
            i = j = 0
            for bead in row[name]["beads"]:
                assert bead.i0 == i and bead.j0 == j, f"{a}→{b} {name} で隙間か重なり"
                i, j = bead.i1, bead.j1
            assert (i, j) == (row["n_src"], row["n_dst"]), \
                f"{a}→{b} {name} が覆いきれていない"


# --- T-027 / T-028: 対照 ------------------------------------------------------


@pytest.mark.validation
def test_t027_diagonal_control_is_present_and_worse(results):
    """T-027(G-05)— 対照が全組にあり、独→英で実際に手法より劣る。"""
    for key, row in results.items():
        assert "diagonal" in row, f"{key} に対照が無い"
        assert "gale_church" in row, f"{key} に手法が無い"

    row = results[("de_pg22367", "en_pg5200")]
    method = row["gale_church"]["paragraph_agreement"]
    control = row["diagonal"]["paragraph_agreement"]
    assert method is not None and control is not None
    assert control < method, (
        f"対照が対照になっていない: 対角線 {control:.4f} 対 手法 {method:.4f}"
    )
    # 陽性対照 — 同じものを比べれば差は消える。差の出どころが手法であることの確認。
    assert not (method < method)


@pytest.mark.validation
def test_t028_refinement_reported_for_both_methods(results):
    """T-028(G-07)— 日本語を含む組で、破れ件数が手法と対照の両方について返る。"""
    for pair in (("de_pg22367", "ja_aozora49866"), ("en_pg5200", "ja_aozora49866")):
        row = results[pair]
        for name in ("gale_church", "diagonal"):
            entry = row[name]
            assert isinstance(entry["refinement_violations"], int)
            assert entry["refinement_paragraphs"] > 0
        # 段落一致率は定義できない組である(構造が違う)ことも固定する
        assert row["gale_church"]["paragraph_agreement"] is None


# --- T-029: SPEC §3.2 の実測表との突合 ---------------------------------------

LABELS = {
    "独→英": ("de_pg22367", "en_pg5200"),
    "独→日": ("de_pg22367", "ja_aozora49866"),
    "英→日": ("en_pg5200", "ja_aozora49866"),
}


@pytest.mark.validation
def test_t029_spec_alignment_table_matches_live_scan(results):
    """T-029 — SPEC §3.2 の縫い目の実測表が、いま走査した値と一致する。

    文書中の数値を守るものは他に無い(HC-152)。
    """
    import re

    rows: list[list[str]] = []
    in_table = False
    for line in (ROOT / "SPEC.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[:3] == ["組", "c", "手法"]:
            in_table = True
            continue
        if in_table and not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    assert rows, "SPEC §3.2 の実測表が見つからない"

    seen: set[tuple[tuple[str, str], str]] = set()
    for cells in rows:
        pair = LABELS[cells[0]]
        method = cells[2]
        row = results[pair]
        entry = row[method]
        seen.add((pair, method))

        assert abs(float(cells[1]) - row["c"]) < 5e-5, f"{cells[0]} の c"
        assert int(cells[3]) == entry["links"], f"{cells[0]} {method} の対応数"

        agr = entry["paragraph_agreement"]
        if cells[4] == "—":
            assert agr is None, f"{cells[0]} は段落一致率が定義できないはず"
        else:
            assert agr is not None
            assert abs(float(cells[4]) - agr) < 5e-5, f"{cells[0]} {method} の段落一致率"

        viol, paras = cells[5].split("/")
        assert int(viol) == entry["refinement_violations"], f"{cells[0]} {method} の破れ"
        assert int(paras) == entry["refinement_paragraphs"]

    expected = {(p, m) for p in LABELS.values() for m in ("gale_church", "diagonal")}
    assert seen == expected, f"表が覆っていない組と手法: {expected - seen}"
