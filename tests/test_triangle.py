"""三角整合の検査。TEST_SPEC の T-054〜T-056 に対応する(G-06)。

**この物差しは単独では品質を意味しない。** 比例写像は合成もまた比例写像になるので
自明に 1.0 を取る —— そして対角線は、段落一致率でも細分の破れでも三手法中いちばん悪い。
だから検査は「整合率が高いこと」ではなく、**四つ(三手法+帰無)が揃って出ること**と
**自明に取れる手法が実際に満点を取ること**を固定する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import align, evaluate  # noqa: E402


@pytest.fixture(scope="session")
def triangle(results):
    return evaluate.triangle_verdict(results)


# --- T-054: 骨格 --------------------------------------------------------------


@pytest.mark.unit
def test_t054_composition_of_identity_is_consistent():
    """T-054 — 合成が恒等なら整合率は 1 になる(合成データで確かめる)。"""
    n = 6
    ident = [align.Bead(i, i + 1, i, i + 1) for i in range(n)]
    t = evaluate.triangle_consistency(ident, ident, ident, n)
    assert t.compared == n and t.agreed == n and t.rate == 1.0
    assert (t.only_direct, t.only_composed, t.neither) == (0, 0, 0)


@pytest.mark.unit
def test_t054_denominator_excludes_one_sided_sentences():
    """T-054 — 片側にしか行き先が無い文は分母から外れ、件数として返る。"""
    n = 4
    direct = [align.Bead(0, 1, 0, 1), align.Bead(1, 2, 1, 2)]      # 文 0,1 だけ
    first = [align.Bead(0, 1, 0, 1), align.Bead(2, 3, 2, 3)]        # 文 0,2 だけ
    second = [align.Bead(0, 1, 0, 1), align.Bead(2, 3, 2, 3)]
    t = evaluate.triangle_consistency(direct, first, second, n)
    assert t.compared == 1 and t.agreed == 1        # 文 0 だけが両側にある
    assert t.only_direct == 1                        # 文 1 は直接だけ
    assert t.only_composed == 1                      # 文 2 は合成だけ
    assert t.neither == 1                            # 文 3 はどちらも無し
    assert t.compared + t.only_direct + t.only_composed + t.neither == n


@pytest.mark.unit
def test_t054_shuffling_the_second_leg_breaks_agreement():
    """T-054 — 合成の後半を無作為に置換すると整合率が崩れる(陽性対照)。

    これが無いと、`triangle_consistency` が常に 1 を返す実装でも
    上のケースは緑になる。
    """
    n = 40
    direct = [align.Bead(i, i + 1, i, i + 1) for i in range(n)]
    first = [align.Bead(i, i + 1, i, i + 1) for i in range(n)]
    assert evaluate.triangle_consistency(direct, first, direct, n).rate == 1.0

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    shuffled = [align.Bead(i, i + 1, int(perm[i]), int(perm[i]) + 1) for i in range(n)]
    assert evaluate.triangle_consistency(direct, first, shuffled, n).rate < 0.2


# --- T-055: 比例写像は自明に整合する ------------------------------------------


@pytest.mark.validation
def test_t055_all_methods_and_null_are_reported(triangle):
    """T-055(G-06)— 三手法と帰無の四つが揃って返る。"""
    assert set(triangle["methods"]) == set(evaluate.METHODS)
    for m in triangle["methods"]:
        t = triangle[m]
        assert 0.0 <= t.rate <= 1.0
        assert t.compared > 0, f"{m} の分母が 0"
        assert m in triangle["null"], f"{m} の帰無が無い"
        assert len(triangle["null"][m]) == evaluate.NULL_TRIALS


@pytest.mark.validation
def test_t055_proportional_map_is_trivially_consistent(triangle):
    """T-055(G-06)— 対角線は自明に整合し、しかし品質は最下位である。

    **この二つを同じ場所で押さえるのが要点である。** 整合率だけを見て
    「良い手法」と読む道を塞ぐ。
    """
    assert triangle["diagonal"].rate > 0.99, (
        f"比例写像が自明に整合していない({triangle['diagonal'].rate:.4f})"
        " — この物差しの性質が変わったので §3.10 を書き直すこと"
    )


@pytest.mark.validation
def test_t055_trivial_consistency_does_not_mean_quality(triangle, results):
    """T-055 — 整合率が満点の手法が、段落一致率では最下位であること。"""
    de_en = results[("de_pg22367", "en_pg5200")]
    rates = {m: de_en[m]["paragraph_agreement"] for m in triangle["methods"]}
    worst = min(rates, key=rates.get)
    assert worst == "diagonal", f"段落一致率の最下位が {worst}"
    assert triangle["diagonal"].rate >= max(
        triangle[m].rate for m in triangle["methods"]), (
        "自明に整合するはずの手法が最上位でない"
    )


@pytest.mark.validation
def test_t055_null_collapses(triangle):
    """T-055 — 帰無では整合率が崩れる。一致が偶然でないことの確認。"""
    for m in triangle["methods"]:
        nulls = triangle["null"][m]
        assert max(nulls) < 0.05, f"{m} の帰無が崩れていない: {nulls}"
        assert triangle[m].rate > 10 * (sum(nulls) / len(nulls)) or triangle[m].rate > 0.2


# --- T-056: SPEC §3.10 との突合 ----------------------------------------------


@pytest.mark.validation
def test_t056_spec_triangle_table_matches_live_scan(triangle):
    """T-056 — SPEC §3.10 の表が、いま走査した値と一致する(HC-152)。"""
    rows: list[list[str]] = []
    in_table = False
    for line in (ROOT / "SPEC.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[:2] == ["手法", "整合率"]:
            in_table = True
            continue
        if in_table and not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    assert rows, "SPEC §3.10 の三角整合表が見つからない"

    seen = set()
    for cells in rows:
        m = cells[0]
        assert m in triangle["methods"], f"未知の手法: {m}"
        seen.add(m)
        t = triangle[m]
        assert abs(float(cells[1].replace("**", "")) - t.rate) < 5e-5, f"{m} 整合率"
        assert int(cells[2]) == t.compared, f"{m} 比較できた"
        assert int(cells[3]) == t.only_direct, f"{m} 直接のみ"
        assert int(cells[4]) == t.only_composed, f"{m} 合成のみ"
        assert int(cells[5]) == t.neither, f"{m} どちらも無し"
        nulls = triangle["null"][m]
        assert abs(float(cells[6]) - sum(nulls) / len(nulls)) < 5e-5, f"{m} 帰無"
    assert seen == set(triangle["methods"]), f"表が覆っていない手法: {seen}"
