"""文の縫い目を引く。L2 では**長さだけ**を手がかりにする。

**循環の禁止(G-03)。** この関数群が受け取るのは `Sequence[str]` —— 文の本文だけである。
章・段落の境界も、文の識別子も、版の名前も渡さない。だから
「引いた縫い目が段落境界を守るか」を後から測ることに意味がある。
渡してしまえば、それは恒等式になる。

避けようのない漏れが一つだけある: **段落境界は文境界の部分集合である**。
文は段落をまたがないので、段落の切れ目は必ずどこかの文の切れ目でもある。
ただし「どの文境界が段落境界でもあるか」は入力に現れないので、
評価の情報がアライナに渡っているわけではない。SPEC §8 に明記する。

**手法。** Gale & Church (1993) の長さモデルを実装する。
文の長さの比が正規分布に従うと仮定し、(1,1)(1,0)(0,1)(2,1)(1,2)(2,2) の
六つの対応の中から、全体の対数尤度が最大になる**単調な**経路を DP で選ぶ。

パラメータの出所を分けて書く:

- `c`(1 文字あたりの伸縮率)は**本文の総字数の比だけ**から出す。
  アラインメントを使わないので循環しない。実測(2026-09-05):
  独→英 0.9768 / 独→日 0.4675 / 英→日 0.4786。
- `s2`(分散)は Gale & Church (1993) が公表した 6.8 をそのまま使う。
  **この corpus に合わせて当てはめない。** 当てはめるには対応が要り、
  対応こそが測りたいものだからである。当てはめずに悪ければ、それは知見である。
- 対応の事前費用も同論文の値(1-1: 0、2-1/1-2: 230、1-0/0-1: 450、2-2: 440)。

**このモジュールはプロジェクト内の他のモジュールを import しない。** 標準ライブラリだけで
閉じている。段落や章を知っている型に触れる道が構文として存在しないので、
「渡していない」は約束ではなく構造である(T-024 が AST で検査する)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Gale & Church (1993) Table 5。**この corpus に当てはめ直していない。**
PRIOR_COST = {
    (1, 1): 0,
    (2, 1): 230,
    (1, 2): 230,
    (1, 0): 450,
    (0, 1): 450,
    (2, 2): 440,
}
GALE_CHURCH_VARIANCE = 6.8
_BIG = 10 ** 9


@dataclass(frozen=True)
class Bead:
    """対応の一単位。src[i0:i1] が dst[j0:j1] に対応する。"""

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.i1 - self.i0, self.j1 - self.j0)


def stretch_from_totals(src: Sequence[str], dst: Sequence[str]) -> float:
    """`c` を本文の総量だけから推定する。**対応を一切使わない。**"""
    total_src = sum(len(s) for s in src)
    total_dst = sum(len(s) for s in dst)
    if total_src == 0:
        raise ValueError("src の総字数が 0")
    return total_dst / total_src


def _match_cost(len_src: int, len_dst: int, c: float, s2: float) -> float:
    """長さの食い違いを対数尤度の費用に直す。"""
    mean = len_src * c
    variance = len_src * s2
    if variance <= 0:
        variance = s2
    delta = (len_dst - mean) / math.sqrt(variance)
    tail = math.erfc(abs(delta) / math.sqrt(2.0))
    if tail < 1e-300:
        return float(_BIG)
    return -100.0 * math.log(tail)


def align(
    src: Sequence[str],
    dst: Sequence[str],
    *,
    c: float,
    s2: float = GALE_CHURCH_VARIANCE,
) -> list[Bead]:
    """単調な対応を DP で選ぶ。**引数は文の本文だけ**(G-03)。"""
    for name, seq in (("src", src), ("dst", dst)):
        if not isinstance(seq, Sequence) or isinstance(seq, str):
            raise TypeError(f"{name} は文字列の列でなければならない")
        if not all(isinstance(x, str) for x in seq):
            raise TypeError(f"{name} に文字列でない要素がある — G-03(循環の禁止)違反")
    n, m = len(src), len(dst)
    if n == 0 or m == 0:
        raise ValueError("空の列は対応づけられない")

    ls = [len(x) for x in src]
    ld = [len(x) for x in dst]
    # 累積和で (2,1) などの合算長を O(1) で取る
    cs = [0] * (n + 1)
    cd = [0] * (m + 1)
    for i, v in enumerate(ls):
        cs[i + 1] = cs[i] + v
    for j, v in enumerate(ld):
        cd[j + 1] = cd[j] + v

    best = [[float(_BIG)] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0

    ops = tuple(PRIOR_COST.items())
    for i in range(n + 1):
        row = best[i]
        for j in range(m + 1):
            here = row[j]
            if here >= _BIG:
                continue
            for (di, dj), prior in ops:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                length_src = cs[ni] - cs[i]
                length_dst = cd[nj] - cd[j]
                cost = here + prior + _match_cost(length_src, length_dst, c, s2)
                if cost < best[ni][nj]:
                    best[ni][nj] = cost
                    back[ni][nj] = (di, dj)

    if best[n][m] >= _BIG:
        raise ValueError("経路が見つからなかった")

    beads: list[Bead] = []
    i, j = n, m
    while (i, j) != (0, 0):
        step = back[i][j]
        if step is None:
            raise ValueError(f"経路が途切れた: ({i}, {j})")
        di, dj = step
        beads.append(Bead(i - di, i, j - dj, j))
        i, j = i - di, j - dj
    beads.reverse()
    return beads


def align_diagonal(n_src: int, n_dst: int) -> list[Bead]:
    """**対照 A** —— 位置だけで引く縫い目。本文を一切見ない。

    これに勝てない手法は、何も学んでいないのと区別がつかない。
    """
    if n_src <= 0 or n_dst <= 0:
        raise ValueError("空の列は対応づけられない")
    beads: list[Bead] = []
    for i in range(n_src):
        j0 = round(i * n_dst / n_src)
        j1 = round((i + 1) * n_dst / n_src)
        beads.append(Bead(i, i + 1, j0, max(j1, j0 + 1) if j1 <= j0 else j1))
    # 末尾が dst を覆いきるように詰める
    if beads and beads[-1].j1 != n_dst:
        b = beads[-1]
        beads[-1] = Bead(b.i0, b.i1, b.j0, n_dst)
    return beads


def links(beads: Sequence[Bead]) -> list[tuple[int, int]]:
    """対応を (src 文番号, dst 文番号) の組に展開する。

    (1,0) / (0,1) の対応は相手が居ないので組を生まない。
    """
    out: list[tuple[int, int]] = []
    for b in beads:
        for i in range(b.i0, b.i1):
            for j in range(b.j0, b.j1):
                out.append((i, j))
    return out


# --- 埋め込みで引く縫い目 -----------------------------------------------------
#
# **自由なつまみを置かない。** 対応の良し悪しは「たまたま同じくらい似ている組」より
# どれだけ似ているかで測る。基準の似かたは、全対の cos の平均から出す ——
# これは対応を一切使わずに計算できるので循環しない。
# 飛ばす(1-0 / 0-1)は 0 点。だから「偶然の水準を超えない対応」は組まれない。


def chance_similarity(src: np.ndarray, dst: np.ndarray) -> float:
    """無関係な組がどれくらい似て見えるかの水準。**対応を使わずに出す。**"""
    return float(_unit(src).mean(axis=0) @ _unit(dst).mean(axis=0))


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"(文数, 次元) の行列が要る: {x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("長さ 0 のベクトルがある")
    return x / n


def _pair_tables(src: np.ndarray, dst: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """対応の形ごとの類似度表。2 文をまとめるときは埋め込みの平均を取る。"""
    a, b = _unit(src), _unit(dst)
    a2 = _unit(a[:-1] + a[1:]) if len(a) > 1 else np.empty((0, a.shape[1]))
    b2 = _unit(b[:-1] + b[1:]) if len(b) > 1 else np.empty((0, b.shape[1]))
    return {(1, 1): a @ b.T, (2, 1): a2 @ b.T, (1, 2): a @ b2.T, (2, 2): a2 @ b2.T}


def align_embeddings(src: np.ndarray, dst: np.ndarray, *,
                     baseline: float) -> list[Bead]:
    """**引数は埋め込みだけ**(G-03)。本文も段落も識別子も渡らない。

    偶然の水準 `baseline` を超えたぶんの合計が最大になる単調な経路を選ぶ。
    超えない対応は組まれず、その文は飛ばされる —— だから被覆率を併記する。
    """
    tables = _pair_tables(src, dst)
    n, m = len(src), len(dst)
    neg = -math.inf
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    shapes = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 0), (0, 1))

    for i in range(n + 1):
        row = best[i]
        for j in range(m + 1):
            here = row[j]
            if here == neg:
                continue
            for di, dj in shapes:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                gain = 0.0 if (di == 0 or dj == 0) else (
                    float(tables[(di, dj)][i, j]) - baseline)
                score = here + gain
                if score > best[ni][nj]:
                    best[ni][nj] = score
                    back[ni][nj] = (di, dj)

    if best[n][m] == neg:
        raise ValueError("経路が見つからなかった")
    beads: list[Bead] = []
    i, j = n, m
    while (i, j) != (0, 0):
        step = back[i][j]
        if step is None:
            raise ValueError(f"経路が途切れた: ({i}, {j})")
        di, dj = step
        beads.append(Bead(i - di, i, j - dj, j))
        i, j = i - di, j - dj
    beads.reverse()
    return beads


def coverage(beads: Sequence[Bead], n_src: int, n_dst: int) -> tuple[float, float]:
    """相手が付いた文の割合(src 側, dst 側)。飛ばした文は分子に入らない。

    手法どうしを比べるとき、**被覆率が違えば段落一致率の分母も違う**。
    片方だけを見て「一致率が高い」と言えないので、必ず対で出す。
    """
    src_hit = {i for b in beads if b.shape[1] > 0 for i in range(b.i0, b.i1)}
    dst_hit = {j for b in beads if b.shape[0] > 0 for j in range(b.j0, b.j1)}
    return len(src_hit) / n_src, len(dst_hit) / n_dst


def shape_counts(beads: Sequence[Bead]) -> dict[tuple[int, int], int]:
    """対応の形(1-1 / 1-2 …)ごとの件数。結果の読み解きに使う。"""
    out: dict[tuple[int, int], int] = {}
    for b in beads:
        out[b.shape] = out.get(b.shape, 0) + 1
    return out

# このモジュールに `main` は無い。本文を読み込む口を持つと、
# 段落を知っている型への経路ができてしまう(G-03)。
# 実行の入口は pipeline/evaluate.py にある。
