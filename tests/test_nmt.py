"""自作 Transformer の検査。TEST_SPEC の T-049〜T-053 に対応する。

**素で書いた multi-head attention は、順伝播が合っていても逆伝播が合っている
保証がない。** 学習は「それらしく損失が下がる」形で失敗するので、
損失曲線を見ても勾配の誤りには気づけない。だから中心差分と突き合わせる(G-21)。

学習対が**位置だけ**で決まること(SPEC §3.6)も、ここで構造として押さえる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _section(doc: str, heading: str) -> str:
    """見出しから次の同じ深さの見出しまでを切り出す。

    **文書の表は節で特定する。** 先頭の列見出しだけで探すと、
    あとから同じ列を持つ表が増えたときに別の表を読む(HC-220)。
    """
    start = doc.find(heading)
    assert start >= 0, f"{heading} が見つからない"
    depth = heading.split(" ")[0]
    rest = doc[start + len(heading):]
    end = rest.find("\n" + depth + " ")
    return rest if end < 0 else rest[:end]
sys.path.insert(0, str(ROOT))

from pipeline import nmt  # noqa: E402

torch = pytest.importorskip("torch", reason="torch が入っていない")


# --- T-049: 素で書いた attention の順伝播 -------------------------------------


@pytest.mark.unit
def test_t049_multi_head_attention_matches_reference():
    """T-049 — ヘッドごとに素朴に回した参照計算と一致する。

    参照はループで書く。実装はまとめて行列積で回すので、**同じ式を別の書き方で
    確かめている**ことになる。
    """
    torch.manual_seed(0)
    b, tq, tk, d, h = 2, 5, 7, 12, 3
    q = torch.randn(b, tq, d, dtype=torch.double)
    k = torch.randn(b, tk, d, dtype=torch.double)
    v = torch.randn(b, tk, d, dtype=torch.double)
    ws = [torch.randn(d, d, dtype=torch.double) for _ in range(4)]

    got, weights = nmt.multi_head_attention(q, k, v, *ws, h)
    assert got.shape == (b, tq, d)
    assert weights.shape == (b, h, tq, tk)
    assert torch.allclose(weights.sum(-1), torch.ones(b, h, tq, dtype=torch.double))

    # 参照 —— ヘッドを 1 つずつ、明示的なループで
    head = d // h
    qh, kh, vh = (x @ w.T for x, w in zip((q, k, v), ws[:3]))
    ref = torch.zeros(b, tq, d, dtype=torch.double)
    for bi in range(b):
        parts = []
        for hi in range(h):
            sl = slice(hi * head, (hi + 1) * head)
            s = qh[bi, :, sl] @ kh[bi, :, sl].T / head ** 0.5
            w = torch.softmax(s, dim=-1)
            parts.append(w @ vh[bi, :, sl])
            assert torch.allclose(w, weights[bi, hi])
        ref[bi] = torch.cat(parts, dim=-1)
    assert torch.allclose(got, ref @ ws[3].T)


@pytest.mark.unit
def test_t049_mask_hides_positions():
    """T-049 — マスクした位置に注意が向かない。**陽性対照つき。**"""
    torch.manual_seed(1)
    b, tq, tk, d, h = 1, 3, 4, 8, 2
    q = torch.randn(b, tq, d, dtype=torch.double)
    k = torch.randn(b, tk, d, dtype=torch.double)
    v = torch.randn(b, tk, d, dtype=torch.double)
    ws = [torch.randn(d, d, dtype=torch.double) for _ in range(4)]

    mask = torch.zeros(b, 1, tq, tk, dtype=torch.bool)
    mask[..., -1] = True
    _, w = nmt.multi_head_attention(q, k, v, *ws, h, mask)
    assert float(w[..., -1].abs().max()) == 0.0, "マスクした位置に注意が漏れている"
    assert torch.allclose(w.sum(-1), torch.ones(b, h, tq, dtype=torch.double))

    # 陰性対照 — マスクしなければ最後の位置にも注意が向く
    _, w2 = nmt.multi_head_attention(q, k, v, *ws, h, None)
    assert float(w2[..., -1].abs().max()) > 0.0


@pytest.mark.unit
def test_t049_rejects_bad_head_count():
    with pytest.raises(ValueError):
        nmt.multi_head_attention(*(torch.zeros(1, 2, 7, dtype=torch.double),) * 3,
                                 *(torch.zeros(7, 7, dtype=torch.double),) * 4, 2)


# --- T-050: 勾配が数値微分と一致する(G-21)-----------------------------------


def _finite_difference(fn, param, idx, eps: float = 1e-6) -> float:
    """中心差分。**倍精度で、片側でなく中心で取る** —— 片側差分は 1 次の誤差が残る。"""
    with torch.no_grad():
        original = param[idx].item()
        param[idx] = original + eps
        plus = float(fn())
        param[idx] = original - eps
        minus = float(fn())
        param[idx] = original
    return (plus - minus) / (2 * eps)


@pytest.mark.validation
def test_t050_gradients_match_finite_differences():
    """T-050(G-21)— 解析勾配が中心差分と一致する。

    小さな倍精度モデルで、無作為に選んだパラメータ成分について比べる。
    **学習は勾配が間違っていても損失が下がる形で失敗する**ので、
    損失曲線では気づけない。
    """
    torch.manual_seed(0)
    model = nmt.build_model(11, 13, d_model=8, n_heads=2, n_layers=1, d_ff=16,
                            max_len=12, dropout=0.0, seed=0).double()
    model.eval()  # dropout を切る(切らないと差分が雑音に埋もれる)

    src = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    tgt = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    src_pad = src.eq(0)
    tgt_pad = tgt.eq(0)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    def loss():
        logits, _ = model(src, src_pad, tgt[:, :-1], tgt_pad[:, :-1])
        return loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))

    model.zero_grad()
    loss().backward()

    rng = np.random.default_rng(0)
    named = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    assert named, "勾配を持つパラメータが無い"
    # attention の重みを必ず含める —— そこが自前で書いた箇所である
    attn_names = [n for n, _ in named if any(k in n for k in ("sa_", "ca_"))]
    assert attn_names, "自作 attention のパラメータが見つからない"

    checked = 0
    worst = 0.0
    for name, p in named:
        flat = p.data.view(-1)
        grad = p.grad.view(-1)
        picks = rng.choice(flat.numel(), size=min(3, flat.numel()), replace=False)
        for i in picks:
            i = int(i)
            num = _finite_difference(loss, flat, i)
            ana = float(grad[i])
            denom = max(abs(num), abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            checked += 1
    assert checked >= 20, f"確かめた成分が {checked} 個しかない"
    assert worst < 1e-5, f"解析勾配と中心差分の相対誤差の最大が {worst:.3e}"


@pytest.mark.unit
def test_t050_finite_difference_check_has_teeth():
    """T-050 — 勾配をずらしたら落ちること(陽性対照)。

    比較そのものが働いていなければ、上のケースは実装が壊れていても緑になる。
    """
    torch.manual_seed(0)
    x = torch.tensor([1.5], dtype=torch.double, requires_grad=True)

    def f():
        return (x ** 3).sum()

    f().backward()
    num = _finite_difference(f, x.data, 0)
    assert abs(num - float(x.grad[0])) / abs(num) < 1e-6, "陰性対照が崩れている"
    wrong = float(x.grad[0]) * 1.001
    assert abs(num - wrong) / abs(num) > 1e-5, "1e-5 の判定が誤差を見分けられない"


# --- T-051 / T-052: 学習対と点数表 --------------------------------------------


@pytest.mark.unit
def test_t051_windows_depend_only_on_counts():
    """T-051(G-03)— 窓は文の本数だけで決まる。本文も段落も見ない。"""
    w1 = nmt.windows(696, 771, 4)
    w2 = nmt.windows(696, 771, 4)
    assert w1 == w2
    assert len(w1) == 696 - 4 + 1
    # 単調で、両端を覆う
    assert list(w1[0][0]) == [0, 1, 2, 3]
    assert w1[-1][0].stop == 696
    assert w1[-1][1].stop == 771
    for (a, b), (c, d) in zip(w1, w1[1:]):
        assert c.start == a.start + 1
        assert d.start >= b.start
    with pytest.raises(ValueError):
        nmt.windows(0, 10, 4)


@pytest.mark.unit
def test_t051_band_width_is_positional():
    """T-051 — 帯の幅が窓幅から決まり、手法の自由度がその中に収まる。"""
    bw = nmt.band_width(696, 771, 4)
    assert bw > 0
    # 窓幅を広げれば帯も広がる(狭まることはない)
    assert nmt.band_width(696, 771, 8) >= bw


@pytest.mark.unit
def test_t052_attention_scores_stay_inside_the_band():
    """T-052 — 点数表は帯の外でゼロ。**手法が帯に閉じ込められていることの確認。**

    帯の外が非ゼロなら、窓の外の情報がどこかから漏れている。
    """
    from pipeline import align

    n, m, w = 40, 46, 4
    allowed = np.zeros((n, m), dtype=bool)
    for src_r, dst_r in nmt.windows(n, m, w):
        for i in src_r:
            for j in dst_r:
                allowed[i, j] = True
    # 帯の外に点を置いた表は、帯の内側だけを見る限り区別できない —— という
    # 確認ではなく、帯の外がゼロであることを直接見る。合成の点数表で形を固定する。
    scores = np.where(allowed, 0.5, 0.0)
    assert scores[~allowed].max(initial=0.0) == 0.0
    assert allowed.sum() < n * m, "帯が全面を覆っている — 対照が成り立たない"
    beads = align.align_attention(scores, baseline=0.1)
    for i, j in align.links(beads):
        assert allowed[i, j], f"帯の外に対応が引かれた: ({i}, {j})"


# --- T-053: 目玉の判定(G-08)------------------------------------------------


@pytest.mark.validation
def test_t053_headline_verdict_matches_spec():
    """T-053(G-08)— SPEC §3.7 に書いた判定が、いま走らせた結果と一致する。

    **目玉は落ちている。** この検査は「落ちたこと」を固定する ——
    数字が動いたら SPEC のほうが嘘になるので、そのときは両方を直す。
    予測(§3.6)は消さない。
    """
    import re

    from pipeline import evaluate

    if not evaluate.ATTENTION_SCORES.exists():
        pytest.skip("attention の点数表が無い(`python -m pipeline.nmt` で作る)")

    v = evaluate.attention_verdict()
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")

    # **表は節見出しで特定する。** 先頭の列見出しだけで探すと、あとから
    # 同じ列を持つ表が増えたときに別の表を読む —— L15 で実際に起きた
    # (外挿検証の §3.16 に「手法 | 対応 | …」の表を足したら、この検査が
    # そちらを目玉の判定表として拾った)。HC-220 の改訂どおり、節で挟む。
    section = _section(spec, "### 3.7")
    rows = []
    in_table = False
    for line in section.splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[:2] == ["手法", "対応"]:
            in_table = True
            continue
        if in_table and not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    assert rows, "SPEC §3.7 の判定表が見つからない"

    seen = set()
    for cells in rows:
        name = cells[0]
        assert name in v, f"未知の手法: {name}"
        seen.add(name)
        e = v[name]
        assert int(cells[1]) == e["links"], f"{name} の対応数"
        assert abs(float(cells[2]) - e["paragraph_agreement"]) < 5e-5, f"{name} 一致率"
        assert abs(float(cells[3]) - e["paragraph_agreement_common"]) < 5e-5, \
            f"{name} 共通部分の一致率"
        cov = [float(x) for x in cells[4].split("/")]
        assert abs(cov[0] - e["coverage_src"]) < 5e-4, f"{name} 被覆 src"
        assert abs(cov[1] - e["coverage_dst"]) < 5e-4, f"{name} 被覆 dst"
    assert seen == {"attention", "diagonal"}, f"表が覆っていない手法: {seen}"

    # 差と p も文書と突き合わせる
    m = re.search(r"差 \+?(-?\d+\.\d+)、p = (\d+\.\d+)", spec)
    assert m, "SPEC §3.7 の検定の記述が見つからない"
    t = v["test"]
    assert abs(float(m.group(1)) - t.observed) < 5e-5, "差"
    assert abs(float(m.group(2)) - t.p_value) < 0.5e-3, "p 値"

    # **落ちていること自体を固定する。** 通るようになったら SPEC を書き換える番である。
    assert not (t.observed > 0 and t.p_value < 0.01), (
        f"目玉が通った(差 {t.observed:+.4f} / p {t.p_value:.5f})— SPEC §3.7 を書き直すこと"
    )
