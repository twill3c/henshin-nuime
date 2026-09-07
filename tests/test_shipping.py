"""出荷物の検査。TEST_SPEC の T-060 に対応する(G-11)。

**テストが緑であることは、出荷物が作れることを意味しない**(HC-062)。
そして作れたとしても、**配られる木にサーバ関数が紛れていないか**は別の話である。
ここでは `out/` を実際に走査する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out"
PUBLIC_DATA = ROOT / "public" / "data"

pytestmark = pytest.mark.validation


def _need_build() -> None:
    if not OUT.exists():
        pytest.skip("out/ が無い(`npm run build` で作る)")


def test_t060_no_server_functions_in_output():
    """T-060(G-11)— 配られる木にサーバ関数が無い。"""
    _need_build()
    # Vercel の関数はこれらの形で現れる。**名前で探す**ので、
    # 新しい形が増えたらここも増やす必要がある(それは検査の限界として書いておく)。
    markers = ["*.func", "*.nft.json", "functions-manifest.json"]
    found: list[str] = []
    for pattern in markers:
        found += [str(p.relative_to(ROOT)) for p in OUT.rglob(pattern)]
    for name in ("functions", "api"):
        found += [str(p.relative_to(ROOT)) for p in OUT.rglob(name) if p.is_dir()]
    assert not found, f"サーバ関数の痕跡: {found}"

    # 陽性対照 —— 探し方が働いていることを、在るはずのものが見つかることで示す
    assert list(OUT.rglob("*.html")), "html すら見つからない — 走査が働いていない"


def test_t060_every_page_is_static_html():
    """T-060 — 画面が静的な html として書き出されている。"""
    _need_build()
    pages = {p.relative_to(OUT).as_posix() for p in OUT.rglob("index.html")}
    assert "index.html" in pages, "入口が無い"
    assert "yomu/index.html" in pages, "三面の本文が無い"


def test_t060_baked_data_is_shipped_and_identical():
    """T-060 — 焼いたデータが**そのまま**配られている。

    `public/` を経由して `out/` に入るが、途中で変換が挟まっていないことを
    中身の一致で確かめる。片方だけ更新して気づかない、を防ぐ。
    """
    _need_build()
    if not PUBLIC_DATA.exists():
        pytest.skip("public/data が無い(`python -m pipeline.bake` で作る)")
    names = sorted(p.name for p in PUBLIC_DATA.glob("*.json"))
    assert names, "焼いたデータが無い"
    for name in names:
        shipped = OUT / "data" / name
        assert shipped.exists(), f"{name} が配られていない"
        assert shipped.read_bytes() == (PUBLIC_DATA / name).read_bytes(), (
            f"{name} が焼いたものと違う"
        )


def test_t060_manifest_matches_the_pipeline():
    """T-060 — 配られた manifest が、いまのパイプラインの出力と一致する。

    **焼き直さないまま公開面だけ触ると、画面が古いデータを出す。**
    """
    _need_build()
    if not (PUBLIC_DATA / "manifest.json").exists():
        pytest.skip("public/data が無い")
    from pipeline import bake

    fresh = bake.build()["manifest.json"]
    shipped = json.loads((OUT / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert shipped == fresh, "配られた manifest がいまのパイプラインの出力と違う"
