"""置換検定の検査。TEST_SPEC の T-045 / T-046 に対応する(G-19)。

**統計の道具は、壊れていてももっともらしい値を返す。** p 値が 0.03 と出たとき、
それが正しいかどうかは値を見ても分からない。だから別経路で確かめる ——
ブロック数が少ない場合は入れ替えの通り数が有限なので、**全数列挙で厳密な p が出せる**。
Monte Carlo の値がそれと合うかを見る。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import stats  # noqa: E402


def _mc_tolerance(p: float, n_iter: int) -> float:
    """Monte Carlo の標準誤差から許容幅を出す。定数で決め打ちしない。"""
    se = math.sqrt(max(p * (1.0 - p), 1e-6) / n_iter)
    return 4.0 * se + 1.0 / (n_iter + 1)


# --- T-045: 二経路照合 --------------------------------------------------------


@pytest.mark.validation
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t045_monte_carlo_matches_exact_enumeration(seed):
    """T-045(G-19)— Monte Carlo の p が全数列挙の厳密値と一致する。"""
    rng = np.random.default_rng(seed)
    # 12 ブロック × 各 3 単位。全数列挙は 2^12 = 4,096 通りで済む。
    blocks = [i // 3 for i in range(36)]
    a = rng.integers(0, 2, size=36).astype(float)
    b = rng.integers(0, 2, size=36).astype(float)

    exact = stats.paired_permutation_exact(a, b, blocks)
    mc = stats.paired_permutation(a, b, blocks, n_iter=20000, seed=seed)

    assert exact.exact and not mc.exact
    assert exact.blocks == mc.blocks == 12
    assert abs(exact.observed - mc.observed) < 1e-12, "観測した差が経路で食い違う"
    tol = _mc_tolerance(exact.p_value, mc.iterations)
    assert abs(exact.p_value - mc.p_value) < tol, (
        f"厳密 {exact.p_value:.5f} 対 Monte Carlo {mc.p_value:.5f}(許容 {tol:.5f})"
    )


@pytest.mark.validation
def test_t045_positive_and_negative_controls():
    """T-045 — 差が無ければ p は 1 に近く、差が大きければ小さくなる。

    これが無いと、`p_value` が常に同じ値を返す実装でも上のケースは緑になりうる。
    """
    blocks = [i // 2 for i in range(24)]

    same = [1.0] * 24
    r = stats.paired_permutation_exact(same, list(same), blocks)
    assert r.observed == 0.0
    assert r.p_value == 1.0, "差 0 なのに p が 1 でない"

    # 全ブロックで a が b を上回る —— 入れ替えても符号が揃うのは 1 通りだけ
    hi, lo = [1.0] * 24, [0.0] * 24
    r = stats.paired_permutation_exact(hi, lo, blocks)
    assert r.observed == 1.0
    assert r.p_value == pytest.approx(2 / 2 ** 12), (
        f"両側 p が {r.p_value} — 全同符号の 2 通りだけのはず"
    )


# --- T-046: 検定器の性質 ------------------------------------------------------


@pytest.mark.unit
def test_t046_blocking_actually_changes_resolution():
    """T-046 — ブロックのまとめ方が p の分解能に効く。

    ブロックが 1 個しかなければ入れ替えは 2 通りしかなく、両側 p は 1 になる。
    ブロックを分けるほど分解能が上がる。**この性質が無いなら、ブロックを
    分けている意味が無い。**
    """
    values_a = [1.0] * 16
    values_b = [0.0] * 16

    one = stats.paired_permutation_exact(values_a, values_b, [0] * 16)
    assert one.blocks == 1
    assert one.p_value == 1.0, "1 ブロックなら両側 p は 1 になるはず"

    many = stats.paired_permutation_exact(values_a, values_b, list(range(16)))
    assert many.blocks == 16
    assert many.p_value < one.p_value, "ブロックを分けても分解能が上がらない"


@pytest.mark.unit
def test_t046_rejects_malformed_input():
    """T-046 — 長さ違い・空入力は黙って通さない。"""
    with pytest.raises(ValueError):
        stats.paired_permutation([1.0, 0.0], [1.0], [0, 0])
    with pytest.raises(ValueError):
        stats.paired_permutation([], [], [])
    with pytest.raises(ValueError):
        stats.paired_permutation_exact([1.0] * 40, [0.0] * 40, list(range(40)),
                                       max_blocks=20)


@pytest.mark.unit
def test_t046_p_display_never_writes_zero():
    """T-046 — 分解能を下回る p を「0」と書かない。

    Monte Carlo で 10,000 回振って一度も超えなければ p は 1/10001 になるが、
    それは「0」ではなく「これ以上細かくは分からない」である。
    """
    blocks = list(range(20))
    r = stats.paired_permutation([1.0] * 20, [0.0] * 20, blocks,
                                 n_iter=10000, seed=0)
    assert r.p_value > 0.0
    assert r.p_display.startswith("< "), f"p_display が {r.p_display}"
    assert "0.00010" in r.p_display

    # 分解能より大きい p はそのまま数字で出る
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2, size=20).astype(float)
    b = rng.integers(0, 2, size=20).astype(float)
    r2 = stats.paired_permutation(a, b, blocks, n_iter=10000, seed=0)
    if r2.p_value > 1.0 / 10001:
        assert not r2.p_display.startswith("< ")


@pytest.mark.unit
def test_t046_result_is_reproducible():
    """T-046 — 同じ種なら同じ p が出る。SPEC に書いた数値が動かない前提。"""
    blocks = [i // 4 for i in range(40)]
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, size=40).astype(float)
    b = rng.integers(0, 2, size=40).astype(float)
    first = stats.paired_permutation(a, b, blocks, seed=0)
    second = stats.paired_permutation(a, b, blocks, seed=0)
    assert first.p_value == second.p_value
    other = stats.paired_permutation(a, b, blocks, seed=1)
    assert other.observed == first.observed  # 観測値は種に依存しない
