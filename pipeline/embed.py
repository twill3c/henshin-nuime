"""文を埋め込む。**二つの独立した経路**で同じことをやり、突き合わせる(G-04)。

  経路A  onnxruntime が Xenova の ONNX 書き出しを実行する
  経路B  この場で書いた NumPy の順伝播が、本家 safetensors の重みを読んで走る

同じ重みなので**一致するのが正しい**。一致しなければ、どちらかの実装が間違っている。
片方(B)はこのプロジェクトが書いたものなので、これは実装訓練の答え合わせでもある。

**完全一致は原理的に取れない**(HC-073)。ONNX Runtime は演算を融合し、
加算の順序も並列度に応じて変わるので、浮動小数の丸めが揃わない。だから閾値は
「一致する量」(どの値を比べるか)と「揃える規則」(dtype・語彙・切り詰め長)に
分けて書き、実測した差を SPEC に置く。

**比べるのは結論だけではない**(HC-065)。プールした 384 次元のベクトルだけを見ると、
経路の途中が違っていても平均で打ち消し合うことがある。だから
**トークンごとの最終層出力**(系列長 × 384)を比べる。

モデル: paraphrase-multilingual-MiniLM-L12-v2(BERT 12 層・隠れ 384・12 ヘッド・
語彙 250,037・絶対位置埋め込み・gelu・LayerNorm eps 1e-12)。
出力は平均プーリングのみで、L2 正規化は**しない**(本家 modules.json に Normalize が無い)。
系列長は本家の `sentence_bert_config.json` に従い 128 で切る。
"""

from __future__ import annotations

import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

if __package__:
    from . import sentences
else:  # スクリプトとして直接起動されたとき(HC-174)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import sentences

MODEL_DIR = (Path(__file__).resolve().parent.parent / "models"
             / "paraphrase-multilingual-MiniLM-L12-v2")
MAX_SEQ_LEN = 128  # 本家 sentence_bert_config.json の max_seq_length

_ERF = np.frompyfunc(math.erf, 1, 1)


class ModelMissing(RuntimeError):
    """重みが手元に無いときに投げる。落とし方は README に書く。"""


# --- safetensors を自前で読む ------------------------------------------------

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64,
              "I64": np.int64, "I32": np.int32, "U8": np.uint8, "BOOL": np.bool_}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """safetensors を読む。書式は「8 バイトの見出し長 + JSON の見出し + 生データ」。

    依存を増やさずに済ませている。読み違えれば経路 B の出力が経路 A と合わなくなるので、
    この関数の正しさは G-04 の照合そのものが担保する。
    """
    raw = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    body = 8 + header_len
    out: dict[str, np.ndarray] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = _ST_DTYPES.get(spec["dtype"])
        if dtype is None:
            raise ValueError(f"未知の dtype: {spec['dtype']}")
        start, end = spec["data_offsets"]
        buf = raw[body + start : body + end]
        out[name] = np.frombuffer(buf, dtype=dtype).reshape(spec["shape"])
    if not out:
        raise ValueError("テンソルが 1 つも読めなかった")
    return out


# --- 経路B: 自前の NumPy 順伝播 ----------------------------------------------


def gelu(x: np.ndarray) -> np.ndarray:
    """厳密な gelu(誤差関数を使う版)。HF の `hidden_act="gelu"` はこれ。

    近似版(tanh)ではない。近似版を使うと経路 A と 1e-3 規模でずれる。
    """
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)).astype(x.dtype))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


@dataclass
class BertConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_eps: float

    @classmethod
    def load(cls, path: Path) -> "BertConfig":
        c = json.loads(path.read_text(encoding="utf-8"))
        return cls(c["hidden_size"], c["num_hidden_layers"],
                   c["num_attention_heads"], c["layer_norm_eps"])


class NumpyEncoder:
    """BERT の順伝播を素で書いたもの。`nn.Transformer` の類は使わない。"""

    def __init__(self, weights: dict[str, np.ndarray], config: BertConfig):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.cfg = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim * config.num_attention_heads != config.hidden_size:
            raise ValueError("隠れ次元がヘッド数で割り切れない")

    def _p(self, name: str) -> np.ndarray:
        try:
            return self.w[name]
        except KeyError as e:  # 重みの名前が変わったら黙って別の結果を出さない
            raise KeyError(f"重みが見つからない: {name}") from e

    def _linear(self, x: np.ndarray, prefix: str) -> np.ndarray:
        # PyTorch の Linear は (out, in) で持つので転置して掛ける
        return x @ self._p(f"{prefix}.weight").T + self._p(f"{prefix}.bias")

    def hidden_states(self, input_ids: np.ndarray,
                      attention_mask: np.ndarray) -> np.ndarray:
        b, t = input_ids.shape
        cfg = self.cfg
        pos = np.arange(t, dtype=np.int64)[None, :].repeat(b, axis=0)
        h = (self._p("embeddings.word_embeddings.weight")[input_ids]
             + self._p("embeddings.position_embeddings.weight")[pos]
             + self._p("embeddings.token_type_embeddings.weight")[0])
        h = layer_norm(h, self._p("embeddings.LayerNorm.weight"),
                       self._p("embeddings.LayerNorm.bias"), cfg.layer_norm_eps)

        # 埋め草のトークンには注意を向けない。加算バイアスで表す。
        bias = (1.0 - attention_mask[:, None, None, :].astype(np.float32)) * -1e4

        for i in range(cfg.num_hidden_layers):
            p = f"encoder.layer.{i}"
            q = self._linear(h, f"{p}.attention.self.query")
            k = self._linear(h, f"{p}.attention.self.key")
            v = self._linear(h, f"{p}.attention.self.value")
            shape = (b, t, cfg.num_attention_heads, self.head_dim)
            q = q.reshape(shape).transpose(0, 2, 1, 3)
            k = k.reshape(shape).transpose(0, 2, 1, 3)
            v = v.reshape(shape).transpose(0, 2, 1, 3)

            scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim) + bias
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)

            ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(b, t, cfg.hidden_size)
            attn = self._linear(ctx, f"{p}.attention.output.dense")
            h = layer_norm(attn + h, self._p(f"{p}.attention.output.LayerNorm.weight"),
                           self._p(f"{p}.attention.output.LayerNorm.bias"),
                           cfg.layer_norm_eps)

            inter = gelu(self._linear(h, f"{p}.intermediate.dense"))
            out = self._linear(inter, f"{p}.output.dense")
            h = layer_norm(out + h, self._p(f"{p}.output.LayerNorm.weight"),
                           self._p(f"{p}.output.LayerNorm.bias"), cfg.layer_norm_eps)
        return h


# --- 経路A: onnxruntime ------------------------------------------------------


class OnnxEncoder:
    def __init__(self, path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # 加算順のばらつきを減らす
        self.session = ort.InferenceSession(str(path), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

    def hidden_states(self, input_ids: np.ndarray,
                      attention_mask: np.ndarray) -> np.ndarray:
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        feed = {k: v for k, v in feed.items() if k in self.input_names}
        missing = set(self.input_names) - set(feed)
        if missing:
            raise ValueError(f"ONNX が要求する入力が足りない: {sorted(missing)}")
        return np.asarray(self.session.run(None, feed)[0], dtype=np.float32)


# --- 共通 --------------------------------------------------------------------


def mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """埋め草を除いた平均。本家の Pooling(mean)と同じ。L2 正規化はしない。"""
    m = attention_mask[..., None].astype(np.float32)
    return (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)


def load_tokenizer(model_dir: Path = MODEL_DIR):
    from tokenizers import Tokenizer

    path = model_dir / "tokenizer.json"
    if not path.exists():
        raise ModelMissing(f"{path} が無い。README の入手手順を見ること")
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    tok.enable_padding(length=None, pad_id=1, pad_token="<pad>")
    return tok


def encode_batch(tok, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    enc = tok.encode_batch(list(texts))
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    return ids, mask


def raw_tokenizer(model_dir: Path = MODEL_DIR):
    """**埋め草も切り詰めも掛けない**トークナイザ。長さを測るためだけに使う。

    `load_tokenizer` が返すものは padding と truncation が有効なので、
    それで長さを測ると**全系列が 128 に揃った値**が返る。
    「128 を超えたものは 0 件」というもっともらしい嘘がそこから出た(HC-174)。

    **`tokenizer.json` 自体に truncation(max_length 128)と padding が焼かれている。**
    読み込んだだけでは素にならないので、明示的に外す。
    """
    from tokenizers import Tokenizer

    path = model_dir / "tokenizer.json"
    if not path.exists():
        raise ModelMissing(f"{path} が無い。README の入手手順を見ること")
    tok = Tokenizer.from_file(str(path))
    tok.no_truncation()
    tok.no_padding()
    return tok


def token_lengths(texts: Iterable[str], model_dir: Path = MODEL_DIR) -> list[int]:
    """**切り詰め前**のトークン数。何件が MAX_SEQ_LEN を超えるかを測る。

    引数にトークナイザを取らない —— 取れるようにすると、
    padding が効いたものを渡せてしまう。
    """
    tok = raw_tokenizer(model_dir)
    if tok.truncation is not None or tok.padding is not None:
        raise ValueError("長さの測定に埋め草・切り詰め付きのトークナイザを使っている")
    return [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=True)]


def _require(path: Path) -> Path:
    if not path.exists():
        raise ModelMissing(f"{path} が無い。README の入手手順を見ること")
    return path


def numpy_encoder(model_dir: Path = MODEL_DIR) -> NumpyEncoder:
    weights = read_safetensors(_require(model_dir / "model.safetensors"))
    return NumpyEncoder(weights, BertConfig.load(_require(model_dir / "config.json")))


def onnx_encoder(model_dir: Path = MODEL_DIR) -> OnnxEncoder:
    return OnnxEncoder(_require(model_dir / "model.onnx"))


EMB_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"


def embed_texts(texts: Sequence[str], *, batch_size: int = 32,
                model_dir: Path = MODEL_DIR) -> np.ndarray:
    """経路 A(onnxruntime)で平均プーリング済みの埋め込みを作る。

    経路 B は 1 文あたり経路 A の 2.5 倍かかるので、全文には使わない。
    B の役目は照合(G-04)であって量産ではない。
    """
    tok = load_tokenizer(model_dir)
    enc = onnx_encoder(model_dir)
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i : i + batch_size])
        ids, mask = encode_batch(tok, chunk)
        out.append(mean_pool(enc.hidden_states(ids, mask), mask))
    return np.concatenate(out, axis=0).astype(np.float32)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cached() -> dict[str, np.ndarray]:
    """焼いた埋め込みを読む。無ければ落ちる(黙って作り直さない)。"""
    manifest = EMB_DIR / "manifest.json"
    if not manifest.exists():
        raise ModelMissing(
            f"{manifest} が無い。`python -m pipeline.embed` で作ること(約 16 分)"
        )
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    return {eid: np.load(EMB_DIR / f"{eid}.npy") for eid in meta["editions"]}


def print_token_stats() -> None:
    """トークン数の分布だけを見る。埋め込みは作らないので数秒で終わる。

    起動経路の検査(T-036)はこちらを使う。全文の埋め込みは 16 分かかり、
    **検査に 16 分かかる工程を入れると、その検査は回されなくなる**。
    """
    for eid, v in sentences.load_all().items():
        lens = token_lengths([s.text for s in v])
        over = sum(1 for x in lens if x > MAX_SEQ_LEN)
        print(f"{eid}: 文 {len(lens)} 件 トークン数 中央値 {int(np.median(lens))} "
              f"最大 {max(lens)} / {MAX_SEQ_LEN} 超 {over} 件 ({over/len(lens):.1%})")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--stats"]:
        print_token_stats()
        return
    if args:
        raise SystemExit(f"使い方: python -m pipeline.embed [--stats] (受け取った: {args})")

    import onnxruntime as ort

    tok = load_tokenizer()
    sents = sentences.load_all()
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    meta: dict[str, object] = {
        "model_dir": MODEL_DIR.name,
        "max_seq_len": MAX_SEQ_LEN,
        "onnxruntime": ort.__version__,
        "intra_op_num_threads": 1,
        "pooling": "mean, no L2 normalisation",
        "sha256": {p.name: _sha256(p)
                   for p in (MODEL_DIR / "model.onnx", MODEL_DIR / "model.safetensors",
                             MODEL_DIR / "tokenizer.json")},
        "editions": {},
    }
    for eid, v in sents.items():
        texts = [s.text for s in v]
        lens = token_lengths(texts)
        over = sum(1 for x in lens if x > MAX_SEQ_LEN)
        vecs = embed_texts(texts)
        np.save(EMB_DIR / f"{eid}.npy", vecs)
        meta["editions"][eid] = {
            "sentences": len(texts),
            "dim": int(vecs.shape[1]),
            "token_len_median": int(np.median(lens)),
            "token_len_max": int(max(lens)),
            "truncated": over,
        }
        print(f"{eid}: 文 {len(texts)} 件 → {vecs.shape}  "
              f"トークン数 中央値 {int(np.median(lens))} 最大 {max(lens)} / "
              f"{MAX_SEQ_LEN} 超 {over} 件 ({over/len(texts):.1%})")
    (EMB_DIR / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {EMB_DIR}")


if __name__ == "__main__":
    main()
