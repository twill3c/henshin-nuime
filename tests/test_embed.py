"""埋め込みの二実装照合。TEST_SPEC の T-031〜T-035 に対応する(G-04)。

同じ重みを、二つの独立した経路で走らせる:

  経路A  onnxruntime が Xenova の ONNX 書き出しを実行する
  経路B  このプロジェクトが書いた NumPy の順伝播が safetensors を読んで走る

**完全一致は取れない**(HC-073)。ONNX Runtime は演算を融合し加算順も変わるので、
float32 の丸めが揃わない。だから閾値は実測してから置く。2026-09-05 の実測:

  二経路の差(トークン単位・埋め草を除く)  最大 4.768e-06
  重みを 1 要素 +0.05 ずらしたときの差      最大 9.156e-04

閾値 1e-4 はこの二つの間にある。上は約 21 倍、下は約 9 倍の余裕がある。
**余裕は大きくない**ので、閾値を緩めるときは陽性対照(T-032)も一緒に見ること。

モデルは約 940 MB あり git に入れていない。手元に無ければ検査は skip する。
入手手順は README にある。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import embed, sentences  # noqa: E402

# 実測(2026-09-05)から置いた閾値。上下の実測値は本モジュールの docstring にある。
AGREEMENT_THRESHOLD = 1e-4


def _need(*names: str) -> None:
    for n in names:
        if not (embed.MODEL_DIR / n).exists():
            pytest.skip(f"モデル {n} が手元に無い(README の入手手順を見ること)")


@pytest.fixture(scope="session")
def sample_batch():
    """三言語から長短が混ざるように少しだけ取る。経路 B が遅いので 9 文。"""
    _need("tokenizer.json")
    tok = embed.load_tokenizer()
    sents = sentences.load_all()
    texts: list[str] = []
    for v in sents.values():
        xs = sorted(v, key=lambda s: len(s.text))
        texts += [xs[i].text for i in np.linspace(0, len(xs) - 1, 3).astype(int)]
    ids, mask = embed.encode_batch(tok, texts)
    return texts, ids, mask


@pytest.fixture(scope="session")
def route_a(sample_batch):
    _need("model.onnx")
    _, ids, mask = sample_batch
    return embed.onnx_encoder().hidden_states(ids, mask)


@pytest.fixture(scope="session")
def weights():
    _need("model.safetensors")
    return embed.read_safetensors(embed.MODEL_DIR / "model.safetensors")


@pytest.fixture(scope="session")
def route_b(sample_batch, weights):
    _need("config.json")
    _, ids, mask = sample_batch
    cfg = embed.BertConfig.load(embed.MODEL_DIR / "config.json")
    return embed.NumpyEncoder(weights, cfg).hidden_states(ids, mask)


def _max_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """埋め草の位置は比べない。どちらの経路もそこを定義していない。"""
    m = mask.astype(bool)
    return float(np.abs(a[m] - b[m]).max())


# --- T-031: 二実装照合 --------------------------------------------------------


@pytest.mark.validation
def test_t031_two_routes_agree(sample_batch, route_a, route_b):
    """T-031(G-04)— トークンごとの最終層出力が閾値以内で一致する。"""
    _, ids, mask = sample_batch
    assert route_a.shape == route_b.shape
    assert mask.sum() > 0, "埋め草でない位置が無い — 走査対象が空"

    diff = _max_diff(route_a, route_b, mask)
    assert diff < AGREEMENT_THRESHOLD, f"二経路の最大差 {diff:.3e}"

    # プール後も一致すること(結論の側)。経路の側は上で見ている。
    pa = embed.mean_pool(route_a, mask)
    pb = embed.mean_pool(route_b, mask)
    cos = (pa * pb).sum(1) / (np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1))
    assert cos.min() > 1 - 1e-5, f"プール後の cos 最小 {cos.min()}"


# --- T-032: 陽性対照 ----------------------------------------------------------


@pytest.mark.validation
def test_t032_perturbing_one_weight_breaks_agreement(sample_batch, route_a, weights):
    """T-032(G-04)— 重みを 1 要素ずらすと照合が落ちる。

    これが無いと、T-031 の緑は「二つの実装が合っている」とも
    「照合が何も見ていない」とも読める。
    """
    _, ids, mask = sample_batch
    cfg = embed.BertConfig.load(embed.MODEL_DIR / "config.json")

    intact = embed.NumpyEncoder(weights, cfg).hidden_states(ids, mask)
    assert _max_diff(route_a, intact, mask) < AGREEMENT_THRESHOLD, "陰性対照が崩れている"

    key = "encoder.layer.5.attention.self.query.weight"
    broken = {k: np.array(v, dtype=np.float32, copy=True) for k, v in weights.items()}
    assert key in broken, f"対照に使う重み {key} が無い"
    broken[key][0, 0] += 0.05

    diff = _max_diff(route_a, embed.NumpyEncoder(broken, cfg).hidden_states(ids, mask),
                     mask)
    assert diff > AGREEMENT_THRESHOLD, (
        f"重みを壊しても差が {diff:.3e} しか出ない — 照合が経路を見ていない"
    )


# --- T-033: safetensors の読み取り -------------------------------------------


@pytest.mark.integration
def test_t033_safetensors_reader(weights):
    """T-033 — 期待するテンソルが読める。壊れた入力では例外になる。"""
    expected_shapes = {
        "embeddings.word_embeddings.weight": (250037, 384),
        "embeddings.position_embeddings.weight": (512, 384),
        "encoder.layer.0.attention.self.query.weight": (384, 384),
        "encoder.layer.11.output.dense.weight": (384, 1536),
    }
    for name, shape in expected_shapes.items():
        assert name in weights, f"{name} が読めていない"
        assert weights[name].shape == shape, f"{name} の形が {weights[name].shape}"
        assert weights[name].dtype == np.float32

    # 12 層ぶんの重みが揃っていること(層番号の取りこぼしを見る)
    layers = {int(k.split(".")[2]) for k in weights if k.startswith("encoder.layer.")}
    assert layers == set(range(12)), f"層が {sorted(layers)}"

    # 陽性対照 — 見出し長を壊した入力では例外になる
    import struct
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.safetensors"
        bad.write_bytes(struct.pack("<Q", 10 ** 9) + b"{}")
        with pytest.raises(Exception):
            embed.read_safetensors(bad)


# --- T-034: 素の演算 ----------------------------------------------------------


@pytest.mark.unit
def test_t034_gelu_is_the_erf_version():
    """T-034 — gelu が誤差関数版であること。tanh 近似版とは実際に食い違う。"""
    import math

    x = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=np.float32)
    exact = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x],
                     dtype=np.float32)
    got = embed.gelu(x)
    assert np.abs(got - exact).max() < 1e-6

    # 近似版とは食い違う。食い違わないなら、この検査は何も言っていない。
    tanh_approx = 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    assert np.abs(got - tanh_approx).max() > 1e-4, "二つの版が区別できていない"


@pytest.mark.unit
def test_t034_layer_norm_matches_closed_form():
    """T-034 — layer_norm が定義どおりであること。"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 8)).astype(np.float32)
    w = rng.normal(size=(8,)).astype(np.float32)
    b = rng.normal(size=(8,)).astype(np.float32)
    got = embed.layer_norm(x, w, b, 1e-12)

    mu = x.mean(-1, keepdims=True)
    sd = np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-12)
    assert np.abs(got - ((x - mu) / sd * w + b)).max() < 1e-5
    # 正規化された軸の平均が 0・分散が 1 になっていること(重みを外して確認)
    z = embed.layer_norm(x, np.ones(8, np.float32), np.zeros(8, np.float32), 1e-12)
    assert np.abs(z.mean(-1)).max() < 1e-5
    assert np.abs(z.var(-1) - 1.0).max() < 1e-3


# --- T-035: キャッシュの整合 --------------------------------------------------


@pytest.mark.validation
def test_t035_embedding_cache_matches_manifest():
    """T-035 — 焼いた埋め込みが、文の数とモデルの実体に対して整合している。"""
    manifest = embed.EMB_DIR / "manifest.json"
    if not manifest.exists():
        pytest.skip("埋め込みキャッシュが無い(`-m pipeline.embed` で作る)")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    sents = sentences.load_all()

    assert set(meta["editions"]) == set(sents), "版の集合が食い違う"
    vecs = embed.load_cached()
    for eid, info in meta["editions"].items():
        assert info["sentences"] == len(sents[eid]), f"{eid} の文数"
        assert vecs[eid].shape == (len(sents[eid]), info["dim"]), f"{eid} の形"
        assert np.isfinite(vecs[eid]).all(), f"{eid} に非有限値"

    stats = meta["editions"]
    assert not all(v["token_len_median"] == embed.MAX_SEQ_LEN for v in stats.values()), (
        "全版でトークン数の中央値が上限に等しい — 埋め草付きで測っている疑い(HC-174)"
    )

    for name, digest in meta["sha256"].items():
        path = embed.MODEL_DIR / name
        if not path.exists():
            pytest.skip(f"モデル {name} が手元に無い")
        assert embed._sha256(path) == digest, f"{name} が記録と違う実体になっている"


# --- T-037: 長さの測定に素のトークナイザを使う -------------------------------


@pytest.mark.validation
def test_t037_token_lengths_use_a_raw_tokenizer():
    """T-037 — 長さの測定が埋め草・切り詰めの影響を受けていないこと。

    `tokenizer.json` には truncation(max_length 128)と padding が**焼かれている**。
    読み込んだだけでは素にならず、そのまま測ると全系列が 128 に揃った値が返り、
    「128 を超えたものは 0 件」というもっともらしい嘘が出る(実際に L3 で出した)。
    """
    _need("tokenizer.json")
    raw = embed.raw_tokenizer()
    assert raw.truncation is None and raw.padding is None, "素になっていない"

    # 読み込んだだけのものは素ではない、という前提そのものを固定する。
    # ここが None になったら、この検査はもう何も守っていない。
    from tokenizers import Tokenizer

    as_shipped = Tokenizer.from_file(str(embed.MODEL_DIR / "tokenizer.json"))
    assert as_shipped.truncation is not None, (
        "配布物に truncation が焼かれていない — この検査の前提が変わった"
    )

    texts = [s.text for s in sentences.load_all()["de_pg22367"]]
    lens = embed.token_lengths(texts)
    assert len(lens) == len(texts)
    assert len(set(lens)) > 1, "長さが全部同じ — 埋め草が効いている"
    assert max(lens) > embed.MAX_SEQ_LEN, (
        "上限を超える文が 1 つも無い — 切り詰めの実測が意味を持たない"
    )

    # 陽性対照 — 埋め草付きのトークナイザで測ろうとしたら止まること
    padded = embed.load_tokenizer()
    assert padded.truncation is not None or padded.padding is not None


# --- T-038: SPEC §3.3 の切り詰め表との突合 -----------------------------------


@pytest.mark.validation
def test_t038_spec_truncation_table_matches_live_scan():
    """T-038 — SPEC §3.3 の切り詰めの表が、いま測った値と一致する(HC-152)。"""
    _need("tokenizer.json")
    rows: list[list[str]] = []
    in_table = False
    for line in (ROOT / "SPEC.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[:2] == ["版", "トークン数 中央値"]:
            in_table = True
            continue
        if in_table and not set("".join(cells)) <= set("-: "):
            rows.append(cells)
    assert rows, "SPEC §3.3 の切り詰め表が見つからない"

    import re

    sents = sentences.load_all()
    seen = set()
    for cells in rows:
        eid = cells[0].strip("`")
        assert eid in sents, f"未知の版: {eid}"
        seen.add(eid)
        lens = np.array(embed.token_lengths([s.text for s in sents[eid]]))
        over = int((lens > embed.MAX_SEQ_LEN).sum())
        lost = int(np.clip(lens - embed.MAX_SEQ_LEN, 0, None).sum())

        assert int(cells[1]) == int(np.median(lens)), f"{eid} 中央値"
        assert int(cells[2]) == int(lens.max()), f"{eid} 最大"
        assert int(re.match(r"(\d+)", cells[3]).group(1)) == over, f"{eid} 128 超"
        got_lost, got_total = (int(x.replace(",", ""))
                               for x in re.findall(r"([\d,]+)", cells[4])[:2])
        assert (got_lost, got_total) == (lost, int(lens.sum())), f"{eid} 落ちるトークン"
    assert seen == set(sents), f"表が覆っていない版: {set(sents) - seen}"


# --- T-039: 焼き直しが再現すること -------------------------------------------


@pytest.mark.validation
def test_t039_embeddings_are_reproducible():
    """T-039 — 同じ入力を二度通すと**ビット単位で**同じ埋め込みが出る。

    `intra_op_num_threads=1` を「揃える規則」として置いている以上(HC-073)、
    再現しなければその規則が効いていない。焼いたファイルどうしを比べる形にすると、
    比較対象が無いときに黙って skip する検査になるので、**その場で二度走らせる**。

    2026-09-06 には全 2,694 文を焼き直して前回とビット単位で一致することも確かめた。
    その記録は `logs/loops/loop_004.jsonl` にある。
    """
    _need("model.onnx", "tokenizer.json")
    texts = [s.text for s in sentences.load_all()["de_pg22367"][:8]]
    first = embed.embed_texts(texts)
    second = embed.embed_texts(texts)
    assert first.shape == (len(texts), 384)
    assert np.array_equal(first, second), (
        f"再現しない(最大差 {np.abs(first - second).max():.3e})"
    )
    # 対照が成り立つ前提 —— 中身が定数なら「一致」は何も言わない
    assert first.std() > 0, "埋め込みが定数になっている"


@pytest.mark.validation
def test_t039_cached_embeddings_match_a_fresh_run():
    """T-039 — 焼いてあるキャッシュが、いま計算し直した値と一致する。

    キャッシュは 16 分かけて作るので普段は作り直さない。だからこそ、
    **その中身が本当にいまのコードの出力なのか**を少数の文で確かめる。
    """
    _need("model.onnx", "tokenizer.json")
    if not (embed.EMB_DIR / "manifest.json").exists():
        pytest.skip("埋め込みキャッシュが無い(`-m pipeline.embed` で作る)")
    cached = embed.load_cached()["de_pg22367"]
    texts = [s.text for s in sentences.load_all()["de_pg22367"][:8]]
    fresh = embed.embed_texts(texts)
    assert np.array_equal(cached[: len(texts)], fresh), (
        f"キャッシュが古い(最大差 {np.abs(cached[:len(texts)] - fresh).max():.3e})"
    )
