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

from pipeline import ingest, sentences  # noqa: E402

SPEC = ROOT / "SPEC.md"
TEST_SPEC = ROOT / "TEST_SPEC.md"

QUOTE_PAIRS = {
    "de_pg22367": "»",
    "en_pg5200": "“",
    "ja_aozora49866": "「",
}


def _table_rows(md: str, *header_cells: str) -> list[list[str]]:
    """指定した列見出しで**始まる**表の本体行を返す。

    先頭 1 列だけで表を特定すると、同じ見出しの別の表を巻き込む —— SPEC には
    「版」で始まる表が三つある(段落の実測・文の実測・規則の寄与)。
    見出しは呼び出し側が必要なだけ並べて特定する。
    """
    n = len(header_cells)
    rows: list[list[str]] = []
    in_table = False
    for line in md.splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if tuple(cells[:n]) == header_cells:
            in_table = True
            continue
        if in_table:
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
    return rows


def _gates_referenced_by_cases() -> set[str]:
    """TEST_SPEC の**ケース表から**参照されているゲート ID。

    前書きでの言及は参照と見なさない —— 散文で `G-08` と書いただけのゲートを
    「守られている」と数えると、この検査は骨抜きになる。
    """
    referenced: set[str] = set()
    for cells in _table_rows(TEST_SPEC.read_text(encoding="utf-8"), "ID", "対応要求"):
        if re.fullmatch(r"T-\d+", cells[0]):
            referenced |= set(re.findall(r"G-\d+", " ".join(cells)))
    return referenced


@pytest.mark.validation
def test_t013_spec_measurements_match_live_scan():
    """T-013 — SPEC §3 の実測表が、いま走査して出た値と一致する。"""
    rows = _table_rows(SPEC.read_text(encoding="utf-8"), "版", "段落")
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
def test_t022_spec_sentence_table_matches_live_scan():
    """T-022 — SPEC §3.1 の文の実測表と規則の寄与表が、いま走査した値と一致する。"""
    spec = SPEC.read_text(encoding="utf-8")
    editions = ingest.load_all()

    rows = _table_rows(spec, "版", "文")
    assert rows, "SPEC §3.1 の文の実測表が見つからない"
    seen = set()
    for cells in rows:
        eid = cells[0].strip("`")
        assert eid in editions, f"未知の版: {eid}"
        seen.add(eid)
        sents = sentences.sentences_for(editions[eid])
        assert int(cells[1].replace(",", "")) == len(sents), f"{eid} 文の総数"
        assert [int(x) for x in re.findall(r"\d+", cells[2])] == \
            sentences.chapter_counts(sents), f"{eid} 章別文数"
        per_para = len(sents) / len(editions[eid].paragraphs)
        assert abs(float(cells[3]) - per_para) < 0.005, f"{eid} 段落あたり"
    assert seen == set(editions), f"覆っていない版: {set(editions) - seen}"

    rows = _table_rows(spec, "版", "規則")
    assert rows, "SPEC §3.1 の規則の寄与表が見つからない"
    label = {"独": "de_pg22367", "英": "en_pg5200", "日": "ja_aozora49866"}
    seen = set()
    for cells in rows:
        eid = label[cells[0]]
        seen.add(eid)
        ed = editions[eid]
        without = sentences.count_for(ed, rules=False)
        with_rules = sentences.count_for(ed, rules=True)
        assert int(cells[2]) == without, f"{eid} 規則を外したときの文数"
        assert int(cells[3]) == without - with_rules, f"{eid} 増分"
    assert seen == set(editions), f"覆っていない版: {set(editions) - seen}"


@pytest.mark.validation
def test_t014_every_gate_is_tested_or_declared_unimplemented():
    """T-014 — SPEC §7 の各 G-xx はテストから参照されるか「未実装」と明記される。"""
    spec = SPEC.read_text(encoding="utf-8")
    gate_rows = _table_rows(spec, "ID", "ゲート")
    gates = {}
    for cells in gate_rows:
        m = re.fullmatch(r"G-\d+", cells[0])
        if not m:
            continue
        gates[cells[0]] = cells[-1]
    assert gates, "SPEC §7 の品質ゲート表が見つからない"

    referenced = _gates_referenced_by_cases()

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


def _gate_loop_claims(spec: str) -> dict[str, str]:
    """ゲート表が書いている「L*n* で実装」を拾う。"""
    out: dict[str, str] = {}
    for cells in _table_rows(spec, "ID", "ゲート"):
        if re.fullmatch(r"G-\d+", cells[0]):
            m = re.search(r"L(\d+) で実装", cells[-1])
            if m:
                out[cells[0]] = f"L{m.group(1)}"
    return out


def _loop_plan_gates(spec: str) -> dict[str, list[int]]:
    """ループ計画がどのループでどのゲートを挙げているか(**複数ありうる**)。

    ゲートは後のループで拡張されることがある —— G-05 と G-07 は L2 で作り
    L4 で埋め込み手法にも広げたので、計画には二度出る。ゲート表が書くのは
    **最初に実装したループ**なので、突合は最小値に対して行う。
    """
    out: dict[str, list[int]] = {}
    for cells in _table_rows(spec, "", "内容"):
        loop = cells[0].strip()
        if not re.fullmatch(r"L\d+", loop):
            continue
        for gate in re.findall(r"G-\d+", cells[2]):
            out.setdefault(gate, []).append(int(loop[1:]))
    return out


@pytest.mark.validation
def test_t044_gate_table_agrees_with_loop_plan():
    """T-044 — ゲート表の「L*n* で実装」と、ループ計画の割り当てが一致する。

    ループを繰り下げると**ゲート表の参照だけが古びる**。L4 の繰り下げで
    実際に 5 箇所ずれた。どちらも人が書く表なので、機械で突き合わせる。
    """
    spec = SPEC.read_text(encoding="utf-8")
    claims = _gate_loop_claims(spec)
    plan = _loop_plan_gates(spec)
    assert claims and plan, "ゲート表かループ計画が読めていない"

    first = {g: f"L{min(v)}" for g, v in plan.items()}
    mismatched = {g: (claims[g], first[g]) for g in claims.keys() & first.keys()
                  if claims[g] != first[g]}
    assert not mismatched, f"ゲート表とループ計画が食い違う: {mismatched}"

    # 計画に載っているゲートは、ゲート表にも存在すること
    assert set(plan) <= set(re.findall(r"G-\d+", spec)), "計画に無いゲートがある"

    # 複数のループにまたがるゲートが実在すること —— この緩和が必要だった証拠
    assert any(len(v) > 1 for v in plan.values()), (
        "またがるゲートが無い — 最小値を取る緩和が何もしていない"
    )

    # 陽性対照 — 番号をずらしたら落ちること
    shifted = {g: f"L{min(v) + 1}" for g, v in plan.items()}
    assert {g for g in claims.keys() & shifted.keys()
            if claims[g] != shifted[g]}, "検査が働いていない"


@pytest.mark.validation
def test_t014_referenced_gates_exist_in_spec():
    """TEST_SPEC が実在しないゲートを参照していないこと(逆向きの検査)。"""
    spec_gates = set(re.findall(r"G-\d+", SPEC.read_text(encoding="utf-8")))
    referenced = _gates_referenced_by_cases()
    assert referenced, "ケース表がゲートを一つも参照していない"
    assert referenced <= spec_gates, f"SPEC に無いゲートを参照: {referenced - spec_gates}"
