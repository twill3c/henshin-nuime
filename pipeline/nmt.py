"""『変身』一冊だけで小さな翻訳器を学習し、その cross-attention を取り出す(F-04 / G-08)。

**multi-head attention は素で書く。** `nn.MultiheadAttention` も
`F.scaled_dot_product_attention` も `nn.Transformer` も使わない —— この企画は
深層学習の実装訓練であり、注意機構の中身を見せることが目的だからである。
自前で書いた勾配が正しいことは、数値微分との照合(G-21)で確かめる。

**学習対の作り方が循環の要である**(SPEC §3.6)。対は**位置だけ**で切る:
独語の連続 W 文の窓と、その位置に比例する英語の窓を対にする。
段落・章・文の識別子・他手法のアラインメントは、入力にも損失にも一切入らない。
モデルが対応について知りうるのは「だいたいこの辺り」だけで、
それは**対照(対角線)が既に持っている情報**である。

したがって attention が対角線を上回れば、上回ったぶんは本文から来たと言える。

**この手法は帯に閉じ込められている。** 窓が位置で決まる以上、独語の文 i と
英語の文 j が同じ窓に入るのは |i·m/n − j| が窓幅の程度に収まるときだけで、
attention の点数はその外側でゼロになる。対角線より**自由度が小さい**手法であって、
大きいのではない。帯の幅は実測して SPEC に書く。
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__:
    from . import align, sentences
else:  # スクリプトとして直接起動されたとき(HC-174)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import align, sentences

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "nmt"

# 位置だけで決める窓。**この値が学習に入る唯一の対応情報である。**
WINDOW = 4
# 偶然の水準。窓の中に独語文が WINDOW 本あるので、注意が均等なら各文のシェアは
# その逆数になる。**窓の構造から決まるので、調整できるつまみではない。**
CHANCE_SHARE = 1.0 / WINDOW
MIN_COUNT = 2  # これ未満の語は <unk> に畳む

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """語と記号に切る。言語ごとに規則を変えない ——
    変えると「どちらの言語で得をしたか」が言えなくなる。"""
    return _TOKEN.findall(text.lower())


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    @classmethod
    def build(cls, texts: Sequence[str], min_count: int = MIN_COUNT) -> "Vocab":
        from collections import Counter

        counts = Counter(t for x in texts for t in tokenize(x))
        kept = sorted(t for t, c in counts.items() if c >= min_count)
        return cls(list(SPECIALS) + kept)


# --- 位置だけで決める窓 -------------------------------------------------------


def windows(n_src: int, n_dst: int, w: int = WINDOW) -> list[tuple[range, range]]:
    """**位置だけ**で対を作る。本文も段落も見ない。

    独語の [i, i+w) に対し、英語は [round(i·m/n), round((i+w)·m/n)) を当てる。
    これは対角線を粗くしたものであり、対照が持っている情報と同じである。
    """
    if n_src <= 0 or n_dst <= 0 or w <= 0:
        raise ValueError("窓を作れない")
    out: list[tuple[range, range]] = []
    for i in range(0, max(n_src - w + 1, 1)):
        hi = min(i + w, n_src)
        j0 = round(i * n_dst / n_src)
        j1 = max(round(hi * n_dst / n_src), j0 + 1)
        out.append((range(i, hi), range(j0, min(j1, n_dst))))
    return out


def band_width(n_src: int, n_dst: int, w: int = WINDOW) -> int:
    """窓が許す対応の帯の幅(英語文の番号で測った最大の広がり)。"""
    widest = 0
    reach: dict[int, list[int]] = {}
    for src_r, dst_r in windows(n_src, n_dst, w):
        for i in src_r:
            reach.setdefault(i, []).extend(dst_r)
    for i, js in reach.items():
        widest = max(widest, max(js) - min(js) + 1)
    return widest


# --- 自作 Transformer ---------------------------------------------------------


def _torch():
    import torch

    return torch


def multi_head_attention(q, k, v, w_q, w_k, w_v, w_o, n_heads: int, mask=None):
    """素で書いた multi-head attention。**注意の重みも一緒に返す。**

    q: (B, Tq, D) / k, v: (B, Tk, D)。mask は (B, 1, Tq, Tk) の bool で、
    True の位置を見えなくする。
    """
    torch = _torch()
    b, tq, d = q.shape
    tk = k.shape[1]
    head = d // n_heads
    if head * n_heads != d:
        raise ValueError("隠れ次元がヘッド数で割り切れない")

    def split(x, w):
        return (x @ w.T).view(b, -1, n_heads, head).transpose(1, 2)

    qh, kh, vh = split(q, w_q), split(k, w_k), split(v, w_v)
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(head)   # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    ctx = (weights @ vh).transpose(1, 2).reshape(b, tq, d)
    return ctx @ w_o.T, weights


def build_model(vocab_src: int, vocab_dst: int, *, d_model: int = 192,
                n_heads: int = 4, n_layers: int = 3, d_ff: int = 384,
                max_len: int = 320, dropout: float = 0.1, seed: int = 0):
    torch = _torch()
    import torch.nn as nn

    torch.manual_seed(seed)

    class Block(nn.Module):
        """自己注意 + 前向き。デコーダでは cross-attention も持つ。"""

        def __init__(self, cross: bool):
            super().__init__()
            self.cross = cross
            for name in ("sa_q", "sa_k", "sa_v", "sa_o"):
                setattr(self, name, nn.Parameter(torch.empty(d_model, d_model)))
                nn.init.xavier_uniform_(getattr(self, name))
            if cross:
                for name in ("ca_q", "ca_k", "ca_v", "ca_o"):
                    setattr(self, name, nn.Parameter(torch.empty(d_model, d_model)))
                    nn.init.xavier_uniform_(getattr(self, name))
                self.ln_ca = nn.LayerNorm(d_model)
            self.ln_sa = nn.LayerNorm(d_model)
            self.ln_ff = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                    nn.Linear(d_ff, d_model))
            self.drop = nn.Dropout(dropout)

        def forward(self, x, memory=None, self_mask=None, cross_mask=None):
            h, _ = multi_head_attention(x, x, x, self.sa_q, self.sa_k, self.sa_v,
                                        self.sa_o, n_heads, self_mask)
            x = self.ln_sa(x + self.drop(h))
            attn = None
            if self.cross:
                h, attn = multi_head_attention(x, memory, memory, self.ca_q,
                                               self.ca_k, self.ca_v, self.ca_o,
                                               n_heads, cross_mask)
                x = self.ln_ca(x + self.drop(h))
            x = self.ln_ff(x + self.drop(self.ff(x)))
            return x, attn

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_src = nn.Embedding(vocab_src, d_model, padding_idx=0)
            self.emb_dst = nn.Embedding(vocab_dst, d_model, padding_idx=0)
            self.pos = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.normal_(self.pos, std=0.02)
            self.enc = nn.ModuleList([Block(False) for _ in range(n_layers)])
            self.dec = nn.ModuleList([Block(True) for _ in range(n_layers)])
            self.out = nn.Linear(d_model, vocab_dst)
            self.drop = nn.Dropout(dropout)
            self.d_model = d_model
            self.n_heads = n_heads

        def encode(self, src, src_pad):
            x = self.drop(self.emb_src(src) * math.sqrt(self.d_model)
                          + self.pos[: src.shape[1]])
            mask = src_pad[:, None, None, :]
            for blk in self.enc:
                x, _ = blk(x, self_mask=mask)
            return x

        def forward(self, src, src_pad, tgt, tgt_pad):
            torch = _torch()
            memory = self.encode(src, src_pad)
            t = tgt.shape[1]
            causal = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
            self_mask = causal[None, None] | tgt_pad[:, None, None, :]
            cross_mask = src_pad[:, None, None, :]
            x = self.drop(self.emb_dst(tgt) * math.sqrt(self.d_model)
                          + self.pos[:t])
            attns = []
            for blk in self.dec:
                x, a = blk(x, memory, self_mask, cross_mask)
                attns.append(a)
            return self.out(x), attns

    return Seq2Seq()


# --- 学習 ---------------------------------------------------------------------


def _batch(seqs: list[list[int]], pad: int, torch):
    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, out.eq(pad)


@dataclass
class Corpus:
    src_tokens: list[list[str]]
    dst_tokens: list[list[str]]
    src_owner: list[list[int]]   # トークンごとの独語文番号
    dst_owner: list[list[int]]   # トークンごとの英語文番号
    v_src: Vocab
    v_dst: Vocab


def build_corpus(src_sents: Sequence[str], dst_sents: Sequence[str],
                 *, w: int = WINDOW) -> Corpus:
    """窓ごとの学習例を作る。**渡すのは文の本文と本数だけ。**"""
    v_src = Vocab.build(src_sents)
    v_dst = Vocab.build(dst_sents)
    st, dt, so, do = [], [], [], []
    for src_r, dst_r in windows(len(src_sents), len(dst_sents), w):
        s_tok, s_own = [], []
        for i in src_r:
            toks = tokenize(src_sents[i])
            s_tok += toks
            s_own += [i] * len(toks)
        d_tok, d_own = [], []
        for j in dst_r:
            toks = tokenize(dst_sents[j])
            d_tok += toks
            d_own += [j] * len(toks)
        if not s_tok or not d_tok:
            continue
        st.append(s_tok)
        dt.append(d_tok)
        so.append(s_own)
        do.append(d_own)
    return Corpus(st, dt, so, do, v_src, v_dst)


def train(corpus: Corpus, *, epochs: int = 30, batch_size: int = 16,
          lr: float = 3e-4, seed: int = 0, log_every: int = 5,
          max_len: int = 416, threads: int | None = None):
    torch = _torch()
    if threads:
        # **スレッド数は結果に影響しうる**(加算順が変わる)。既定は変えず、
        # 明示的に渡されたときだけ上げ、その値を history に残す。
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = build_model(len(corpus.v_src), len(corpus.v_dst), seed=seed,
                        max_len=max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bos, eos, pad = corpus.v_dst.stoi[BOS], corpus.v_dst.stoi[EOS], 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)

    src_ids = [corpus.v_src.encode(t)[: max_len] for t in corpus.src_tokens]
    dst_ids = [[bos] + corpus.v_dst.encode(t)[: max_len - 2] + [eos]
               for t in corpus.dst_tokens]
    order = np.arange(len(src_ids))
    rng = np.random.default_rng(seed)
    history: list[float] = []

    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        total = count = 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            s, s_pad = _batch([src_ids[i] for i in idx], pad, torch)
            d, d_pad = _batch([dst_ids[i] for i in idx], pad, torch)
            logits, _ = model(s, s_pad, d[:, :-1], d_pad[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                           d[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        history.append(total / count)
        if log_every and (ep + 1) % log_every == 0:
            print(f"  epoch {ep + 1:3d}  loss {history[-1]:.4f}")
    return model, history


# --- cross-attention から縫い目の点数表を作る ---------------------------------


def attention_scores(model, corpus: Corpus, n_src: int, n_dst: int, *,
                     max_len: int = 416) -> np.ndarray:
    """cross-attention を文 × 文の点数表に畳む。

    層とヘッドで平均し、英語トークンごとの注意分布を独語の**文**にまとめ、
    さらに英語の**文**で平均する。窓をまたいで足し合わせ、
    最後に「その組が何回同じ窓に入ったか」で割る。

    **英語文のトークン数で割る**のが要である。割らずに足すと、点数は
    「対応の強さ」ではなく「英語文の長さ × 平均シェア」になる —— L6 で実際に
    そうなり、非ゼロの平均 8.25 は英文の語数 35 前後 × 窓内 4 文への均等シェア
    0.25 とほぼ一致していた(HC-174 の系列: もっともらしい数が出るので気づけない)。

    正規化後の値は「英語文 j が独語文 i に向けた注意のシェア」で [0, 1] に収まり、
    偶然の水準は窓の構造から決まる(`CHANCE_SHARE` = 窓内の独語文数の逆数)。
    """
    torch = _torch()
    model.eval()
    bos, eos, pad = corpus.v_dst.stoi[BOS], corpus.v_dst.stoi[EOS], 0
    total = np.zeros((n_src, n_dst), dtype=np.float64)
    seen = np.zeros((n_src, n_dst), dtype=np.float64)

    with torch.no_grad():
        for k in range(len(corpus.src_tokens)):
            s_ids = corpus.v_src.encode(corpus.src_tokens[k])[:max_len]
            s_own = corpus.src_owner[k][:max_len]
            d_ids = ([bos] + corpus.v_dst.encode(corpus.dst_tokens[k])[: max_len - 2]
                     + [eos])
            d_own = corpus.dst_owner[k][: max_len - 2]
            s, s_pad = _batch([s_ids], pad, torch)
            d, d_pad = _batch([d_ids], pad, torch)
            _, attns = model(s, s_pad, d[:, :-1], d_pad[:, :-1])
            # (層, ヘッド, Tq, Tk) を平均して (Tq, Tk)
            a = torch.stack(attns).mean(dim=(0, 2))[0].numpy()
            # 目標側の位置 0 は <bos> なので、1 以降が d_own に対応する
            a = a[1 : 1 + len(d_own)]
            if a.shape[0] == 0:
                continue
            # トークン → 独語文 に畳む(source 側の和)
            src_ids_arr = np.asarray(s_own)
            per_token = np.zeros((a.shape[0], n_src), dtype=np.float64)
            for i in set(s_own):
                per_token[:, i] = a[:, src_ids_arr == i].sum(axis=1)
            # 英語文ごとに**平均**する。和にするとトークン数が点数に混ざる。
            dst_ids_arr = np.asarray(d_own)
            for j in set(d_own):
                rows = per_token[dst_ids_arr == j]
                total[:, j] += rows.mean(axis=0)
            for i in set(s_own):
                for j in set(d_own):
                    seen[i, j] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(seen > 0, total / np.maximum(seen, 1.0), 0.0)
    return scores


def attention_scores_by_head(model, corpus: Corpus, n_src: int, n_dst: int, *,
                             max_len: int = 416) -> np.ndarray:
    """層 × ヘッドごとの点数表を (層, ヘッド, n_src, n_dst) で返す。

    **これは探索用であって判定用ではない。** 事前登録した集約は「層とヘッドで平均」
    (`attention_scores`)であり、結果を見てから良いヘッドを選べば
    それは選び方が結果を作っているだけになる。内訳は所見として出すに留める。
    """
    torch = _torch()
    model.eval()
    bos, pad = corpus.v_dst.stoi[BOS], 0
    n_layers = len(model.dec)
    n_heads = model.n_heads
    total = np.zeros((n_layers, n_heads, n_src, n_dst), dtype=np.float64)
    seen = np.zeros((n_src, n_dst), dtype=np.float64)

    with torch.no_grad():
        for k in range(len(corpus.src_tokens)):
            s_ids = corpus.v_src.encode(corpus.src_tokens[k])[:max_len]
            s_own = np.asarray(corpus.src_owner[k][:max_len])
            d_ids = ([bos] + corpus.v_dst.encode(corpus.dst_tokens[k])[: max_len - 2]
                     + [corpus.v_dst.stoi[EOS]])
            d_own = np.asarray(corpus.dst_owner[k][: max_len - 2])
            s, s_pad = _batch([s_ids], pad, torch)
            d, d_pad = _batch([d_ids], pad, torch)
            _, attns = model(s, s_pad, d[:, :-1], d_pad[:, :-1])
            stacked = torch.stack(attns)[:, 0].numpy()  # (層, ヘッド, Tq, Tk)
            stacked = stacked[:, :, 1 : 1 + len(d_own)]
            if stacked.shape[2] == 0:
                continue
            for i in set(s_own.tolist()):
                col = stacked[:, :, :, s_own == i].sum(axis=3)   # (層, ヘッド, Tq)
                for j in set(d_own.tolist()):
                    total[:, :, i, j] += col[:, :, d_own == j].mean(axis=2)
            for i in set(s_own.tolist()):
                for j in set(d_own.tolist()):
                    seen[i, j] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, total / np.maximum(seen, 1.0), 0.0)


def save_model(model, corpus: Corpus, history: list[float], meta: dict) -> None:
    """重みと語彙を残す。**点数の作り方を直すたびに 45 分学習し直さないため。**"""
    torch = _torch()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    (OUT_DIR / "vocab.json").write_text(
        json.dumps({"src": corpus.v_src.itos, "dst": corpus.v_dst.itos},
                   ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "history.json").write_text(
        json.dumps({"loss": history, **meta}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def load_model(corpus: Corpus, *, max_len: int = 416):
    """保存した重みを読む。語彙が食い違ったら黙って別のものを返さず落ちる。"""
    torch = _torch()
    path = OUT_DIR / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。`python -m pipeline.nmt` で学習する")
    saved = json.loads((OUT_DIR / "vocab.json").read_text(encoding="utf-8"))
    if saved["src"] != corpus.v_src.itos or saved["dst"] != corpus.v_dst.itos:
        raise ValueError("保存時と語彙が違う — 本文か語彙規則が変わっている")
    model = build_model(len(corpus.v_src), len(corpus.v_dst), max_len=max_len)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    scores_only = args == ["--scores-only"]
    if args and not scores_only:
        raise SystemExit(f"使い方: python -m pipeline.nmt [--scores-only](受け取った: {args})")

    sents = sentences.load_all()
    src = [s.text for s in sents["de_pg22367"]]
    dst = [s.text for s in sents["en_pg5200"]]
    print(f"窓幅 {WINDOW} / 帯の幅 {band_width(len(src), len(dst), WINDOW)} 文")
    corpus = build_corpus(src, dst)
    print(f"学習例 {len(corpus.src_tokens)} 件  "
          f"語彙 独 {len(corpus.v_src)} / 英 {len(corpus.v_dst)}")
    threads = 8
    if scores_only:
        _torch().set_num_threads(threads)
        model = load_model(corpus)
        history = json.loads((OUT_DIR / "history.json").read_text(
            encoding="utf-8"))["loss"]
    else:
        model, history = train(corpus, threads=threads)
    meta = {"window": WINDOW, "threads": threads, "chance_share": CHANCE_SHARE,
            "band_width": band_width(len(src), len(dst), WINDOW),
            "vocab_src": len(corpus.v_src), "vocab_dst": len(corpus.v_dst),
            "examples": len(corpus.src_tokens)}
    if not scores_only:
        save_model(model, corpus, history, meta)
    scores = attention_scores(model, corpus, len(src), len(dst))
    np.save(OUT_DIR / "attention_de_en.npy", scores)
    beads = align.align_attention(scores, baseline=CHANCE_SHARE)
    print(f"損失 {history[0]:.4f} → {history[-1]:.4f}  "
          f"点数 最大 {scores.max():.3f} 非ゼロ平均 {scores[scores > 0].mean():.3f}  "
          f"対応 {len(align.links(beads))} 本  → {OUT_DIR}")


if __name__ == "__main__":
    main()
