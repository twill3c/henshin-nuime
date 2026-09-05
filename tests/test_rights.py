"""権利台帳の検査。TEST_SPEC の T-011 / T-012 に対応する(G-01)。

台帳は「こう書いた」だけでは資産にならない。青空文庫の記載事項については、
**生ファイルの奥付と突き合わせる**(二文書突合)。台帳が原本から外れたら落ちる。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import ingest, rights  # noqa: E402


@pytest.fixture(scope="module")
def ledger():
    return rights.load_ledger()


@pytest.mark.unit
def test_t011_ledger_covers_every_edition(ledger):
    """T-011 — 台帳が取り込み対象の三版をちょうど覆う。"""
    assert rights.check_ledger(ledger) == []
    ids = {e["edition_id"] for e in ledger["editions"]}
    assert ids == set(ingest.LOADERS), f"台帳 {ids} と取り込み対象 {set(ingest.LOADERS)} が不一致"


@pytest.mark.integration
def test_t011_aozora_colophon_matches_source(ledger):
    """T-011 — 青空文庫の記載事項が生ファイルの奥付と一致する(二文書突合)。

    青空文庫の取り扱い規準(2026-09-05 取得)は、作品名・著者名・翻訳者名・底本・
    入力者・校正者・日付の情報が削除されないことを求めている。台帳に写した値が
    原本と食い違っていないことを、原本の側から確かめる。
    """
    raw = (ingest.RAW_DIR / "ja_aozora49866.html").read_bytes().decode("shift_jis")
    colophon = raw.split('<div class="bibliographical_information">', 1)[1]
    colophon = colophon.split("</div>", 1)[0]
    assert colophon.strip(), "奥付の抽出に失敗 — 突合の前提が崩れている"

    entry = rights.get(ledger, "ja_aozora49866")
    for field in ("base_text", "publisher", "input_by", "proofread_by",
                  "created", "revised"):
        value = entry[field]
        assert value, f"{field} が空"
        assert value in colophon, f"{field}={value!r} が原本の奥付に無い"


@pytest.mark.unit
def test_t012_ledger_positive_control(ledger):
    """T-012 — 必須欄を落とした台帳は落ちる(陽性対照)。

    検査器が実際に撃つことを確かめないと、T-011 の緑は「台帳が正しい」とも
    「検査が働いていない」とも読める。
    """
    assert rights.check_ledger(ledger) == [], "陰性対照: 正しい台帳は通る"

    for field in ("source_url", "retrieved", "basis", "terms_url"):
        broken = copy.deepcopy(ledger)
        broken["editions"][0][field] = ""
        assert rights.check_ledger(broken), f"{field} を空にしても通ってしまう"

    # 表示義務が空でも落ちること
    broken = copy.deepcopy(ledger)
    broken["editions"][0]["attribution"] = []
    assert rights.check_ledger(broken), "attribution が空でも通ってしまう"

    # 青空版だけに要る欄が欠けても落ちること
    broken = copy.deepcopy(ledger)
    del rights.get(broken, "ja_aozora49866")["base_text"]
    assert rights.check_ledger(broken), "底本が無くても通ってしまう"

    # 版がひとつ欠けても落ちること
    broken = copy.deepcopy(ledger)
    broken["editions"] = broken["editions"][:-1]
    assert rights.check_ledger(broken), "版が欠けても通ってしまう"
