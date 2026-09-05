"""文書の検査。TEST_SPEC の T-013 / T-014 に対応する。

**文書中の数値は、実装のテストでは守られない**(HC-152)。テストは実装を守るが、
SPEC に書いた「段落 97」を守るものは何も無い。だからここで、SPEC の実測表を
生成したのと同じ走査をやり直して突き合わせる。

**品質ゲート表は宣言であって実装ではない**(HC-157)。G-xx を書いた時点で守られていると
錯覚しやすく、書き忘れたゲートについてテストは沈黙する。だから対応そのものを検査する。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import ingest  # noqa: E402

SPEC = ROOT / "SPEC.md"
TEST_SPEC = ROOT / "TEST_SPEC.md"

QUOTE_PAIRS = {
    "de_pg22367": "»",
    "en_pg5200": "“",
    "ja_aozora49866": "「",
}


def _table_rows(md: str, header_cell: str) -> list[list[str]]:
    """`header_cell` を最初の列見出しに持つ表の本体行を返す。"""
    rows: list[list[str]] = []
    in_table = False
    for line in md.splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == header_cell:
            in_table = True
            continue
        if in_table:
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
    return rows


@pytest.mark.validation
def test_t013_spec_measurements_match_live_scan():
    """T-013 — SPEC §3 の実測表が、いま走査して出た値と一致する。"""
    rows = _table_rows(SPEC.read_text(encoding="utf-8"), "版")
    assert rows, "SPEC §3 の実測表が見つからない"

    editions = ingest.load_all()
    seen = set()
    for cells in rows:
        eid = cells[0].strip("`")
        if eid not in editions:
            continue
        seen.add(eid)
        ed = editions[eid]
        text = " ".join(p.text for p in ed.paragraphs)

        doc_total = int(cells[1].replace(",", ""))
        doc_per_ch = [int(x) for x in re.findall(r"\d+", cells[2])]
        doc_chars = int(cells[3].replace(",", ""))
        doc_quotes = int(re.search(r"(\d+)\s*$", cells[4]).group(1))
        doc_quotes_ch = [int(x) for x in re.findall(r"\d+", cells[5])]

        assert doc_total == len(ed.paragraphs), f"{eid} 段落総数"
        assert doc_per_ch == ingest.chapter_counts(ed.paragraphs), f"{eid} 章別段落数"
        assert doc_chars == sum(len(p.text) for p in ed.paragraphs), f"{eid} 本文字数"

        opener = QUOTE_PAIRS[eid]
        assert doc_quotes == text.count(opener), f"{eid} 開き引用符の総数"
        per_ch = [0, 0, 0]
        for p in ed.paragraphs:
            per_ch[p.chapter - 1] += p.text.count(opener)
        assert doc_quotes_ch == per_ch, f"{eid} 章別引用符数"

    assert seen == set(editions), f"SPEC の実測表が覆っていない版: {set(editions) - seen}"


@pytest.mark.validation
def test_t014_every_gate_is_tested_or_declared_unimplemented():
    """T-014 — SPEC §7 の各 G-xx はテストから参照されるか「未実装」と明記される。"""
    spec = SPEC.read_text(encoding="utf-8")
    gate_rows = _table_rows(spec, "ID")
    gates = {}
    for cells in gate_rows:
        m = re.fullmatch(r"G-\d+", cells[0])
        if not m:
            continue
        gates[cells[0]] = cells[-1]
    assert gates, "SPEC §7 の品質ゲート表が見つからない"

    referenced = set(re.findall(r"G-\d+", TEST_SPEC.read_text(encoding="utf-8")))
    # ケース表の外(前書き)での言及は参照と見なさない
    case_rows = _table_rows(TEST_SPEC.read_text(encoding="utf-8"), "ID")
    referenced = set()
    for cells in case_rows:
        if re.fullmatch(r"T-\d+", cells[0]):
            referenced |= set(re.findall(r"G-\d+", " ".join(cells)))

    unguarded = [
        g for g, status in gates.items()
        if g not in referenced and "未実装" not in status
    ]
    assert not unguarded, (
        f"テストからも参照されず「未実装」とも書かれていないゲート: {unguarded}"
    )

    # 陽性対照 — 参照も未実装表記も無いゲートを混ぜたら落ちること
    fake_gates = dict(gates)
    fake_gates["G-99"] = "L0 で実装"
    fake_unguarded = [
        g for g, status in fake_gates.items()
        if g not in referenced and "未実装" not in status
    ]
    assert fake_unguarded == ["G-99"], "検査が働いていない"


@pytest.mark.validation
def test_t014_referenced_gates_exist_in_spec():
    """TEST_SPEC が実在しないゲートを参照していないこと(逆向きの検査)。"""
    spec_gates = set(re.findall(r"G-\d+", SPEC.read_text(encoding="utf-8")))
    case_rows = _table_rows(TEST_SPEC.read_text(encoding="utf-8"), "ID")
    referenced = set()
    for cells in case_rows:
        if re.fullmatch(r"T-\d+", cells[0]):
            referenced |= set(re.findall(r"G-\d+", " ".join(cells)))
    assert referenced, "ケース表がゲートを一つも参照していない"
    assert referenced <= spec_gates, f"SPEC に無いゲートを参照: {referenced - spec_gates}"
