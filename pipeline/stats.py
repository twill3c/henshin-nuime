"""手法間の差が偶然で説明できるかを測る。

**対応のある置換検定**を使う。同じ単位(文・段落)を二つの手法が別々に処理した
結果が手元にあるので、帰無仮説は「**どちらの手法で処理したかは、その単位の
出来不出来に関係ない**」になる。ならば手法の札を入れ替えても差の分布は変わらない。
入れ替えを繰り返して、実際に観測した差がどれくらい端に位置するかを見る。

**並べ替えの単位は段落ブロックにする。** 文どうしは独立でない —— 単調な DP は
隣の文の対応に引きずられるので、文ごとに札を入れ替えると相関を壊して
p 値が甘くなる。同じ段落に属する文はまとめて入れ替える。

**検定器そのものを確かめる。** Monte Carlo の p 値を、ブロック数の少ない場合の
**全数列挙**と突き合わせる(T-046)。統計の道具は「もっともらしい値」を返すので、
合っているかどうかを別の経路で見ないと分からない。

**この節の検定は確認であって、事前登録された予測の検定ではない。** L4 の数値を
見たあとで当てているので、ここで出る p 値は「見えている差が偶然で説明できるか」
までしか言わない。事前登録した予測に当てるのは G-08(L6)である。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    iterations: int
    blocks: int
    exact: bool

    @property
    def p_display(self) -> str:
        """p が分解能を下回ったときに「0」と書かない。"""
        if self.exact:
            return f"{self.p_value:.6f}"
        floor = 1.0 / (self.iterations + 1)
        return f"< {floor:.5f}" if self.p_value <= floor else f"{self.p_value:.5f}"


def _prepare(a: Sequence[float], b: Sequence[float],
             blocks: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    blk = np.asarray(blocks)
    if not (len(a) == len(b) == len(blk)):
        raise ValueError(f"長さが違う: {len(a)} / {len(b)} / {len(blk)}")
    if len(a) == 0:
        raise ValueError("単位が 0 件")
    # ブロックごとの差の合計。札の入れ替えは、この差の符号を反転させることに等しい。
    keys, inverse = np.unique(blk, return_inverse=True)
    diff = np.zeros(len(keys))
    np.add.at(diff, inverse, a - b)
    return diff, np.asarray([np.sum(inverse == i) for i in range(len(keys))],
                            dtype=np.float64)


def paired_permutation(a: Sequence[float], b: Sequence[float],
                       blocks: Sequence[int], *, n_iter: int = 10000,
                       seed: int = 0) -> PermutationResult:
    """手法 a と b の平均の差について、両側 p 値を Monte Carlo で出す。"""
    diff, sizes = _prepare(a, b, blocks)
    total = sizes.sum()
    observed = diff.sum() / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_iter, len(diff)))
    null = (signs * diff).sum(axis=1) / total
    # 観測値自身を分子と分母に入れる(p が 0 にならないようにする標準の作法)
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    p = (extreme + 1) / (n_iter + 1)
    return PermutationResult(observed, p, n_iter, len(diff), exact=False)


def paired_permutation_exact(a: Sequence[float], b: Sequence[float],
                             blocks: Sequence[int], *,
                             max_blocks: int = 20) -> PermutationResult:
    """全数列挙で厳密な両側 p 値を出す。**検定器の照合にだけ使う。**"""
    diff, sizes = _prepare(a, b, blocks)
    n = len(diff)
    if n > max_blocks:
        raise ValueError(f"ブロックが {n} 個 — 全数列挙は {max_blocks} 個まで")
    total = sizes.sum()
    observed = diff.sum() / total
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        if abs(np.dot(signs, diff) / total) >= abs(observed) - 1e-12:
            extreme += 1
    return PermutationResult(observed, extreme / 2 ** n, 2 ** n, n, exact=True)
