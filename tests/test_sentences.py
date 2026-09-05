"""文分割の検査。TEST_SPEC の T-016〜T-021 に対応する。

期待値の中心は「**規則の寄与が、その言語を単独で数えた現象の件数と一致する**」である。
片方は分割器の挙動、もう片方は本文の単純な数え上げで、互いを参照していない。
規則を緩めれば増分が変わり、現象の数え方を間違えれば一致しなくなる。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import ingest, sentences  # noqa: E402


@pytest.fixture(scope="module")
def editions():
    return ingest.load_all()


# --- T-016 / T-017: 取りこぼしと区間の健全性 ---------------------------------


@pytest.mark.integration
def test_t016_split_is_lossless(editions):
    """T-016(G-14)— 文の区間の外に落ちるのは空白だけ。"""
    n = 0
    for ed in editions.values():
        for p in ed.paragraphs:
            sp = sentences.spans(p.text, ed.lang)
            sentences.check_lossless(p.text, sp)  # 例外なら落ちる
            n += len(sp)
    assert n > 0, "文が 1 つも取れていない — 走査対象が空"


@pytest.mark.unit
def test_t016_lossless_positive_control():
    """T-016 — 取りこぼしを人為的に作ると落ちる(陽性対照)。

    `check_lossless` が常に通る実装でも上のケースは緑になるので、撃たせる。
    """
    text = "Erste. Zweite. Dritte."
    good = sentences.spans(text, "de")
    sentences.check_lossless(text, good)  # 陰性対照

    # 本文を 1 文字取りこぼす
    dropped = [good[0], (good[1][0] + 1, good[1][1]), good[2]]
    with pytest.raises(sentences.SplitError):
        sentences.check_lossless(text, dropped)
    # 末尾を落とす
    with pytest.raises(sentences.SplitError):
        sentences.check_lossless(text, good[:-1])
    # 空の区間
    with pytest.raises(sentences.SplitError):
        sentences.check_lossless(text, [(0, 0)])


@pytest.mark.integration
def test_t017_spans_are_ordered_and_disjoint(editions):
    """T-017(G-14)— 区間は単調増加で重ならず、文は空でも前後空白つきでもない。

    **隣接は重なりではない。** 日本語は句点の直後に空白を置かないので、
    区間は必ず `start == prev_end` で接する。禁じるのは `start < prev_end` だけ
    (HC-175 — 禁止形の射程を述語に書き込む)。
    """
    adjacent = 0
    for ed in editions.values():
        for p in ed.paragraphs:
            prev_end = 0
            for start, end in sentences.spans(p.text, ed.lang):
                assert start >= prev_end, f"{p.pid} で区間が重なった"
                if start == prev_end:
                    adjacent += 1
                assert start < end
                s = p.text[start:end]
                assert s == s.strip() and s, f"{p.pid} の文が空か空白つき: {s!r}"
                prev_end = end
    assert adjacent > 0, "隣接が 1 件も無い — この緩和が必要だったことの確認"


# --- T-018〜T-020: 規則の寄与と現象の数え上げの一致(G-15) --------------------


@pytest.mark.validation
def test_t018_de_rule_contribution_matches_phenomena(editions):
    """T-018(G-15)— 独語の例外規則の寄与が、独語だけを数えた現象と一致する。"""
    de = editions["de_pg22367"]
    with_rules = sentences.count_for(de, rules=True)
    without = sentences.count_for(de, rules=False)

    lowercase = hyphen = 0
    for p in de.paragraphs:
        for m in re.finditer(r"[.!?]«?(\s+)(\S)", p.text):
            nxt = m.group(2)
            if nxt.islower():
                lowercase += 1
            elif nxt == "-":
                hyphen += 1
    assert lowercase > 0 and hyphen > 0, "現象が 0 件 — 対照が成り立たない"
    assert without - with_rules == lowercase + hyphen, (
        f"寄与 {without - with_rules} 対 現象 {lowercase}+{hyphen}"
    )


@pytest.mark.validation
def test_t019_en_abbreviation_contribution_matches_occurrences(editions):
    """T-019(G-15)— 英語の略語規則の寄与が `Mr.` / `Mrs.` の出現数と一致する。"""
    en = editions["en_pg5200"]
    with_rules = sentences.count_for(en, rules=True)
    without = sentences.count_for(en, rules=False)

    body = " ".join(p.text for p in en.paragraphs)
    occurrences = sum(
        len(re.findall(r"(?:^|\s)" + re.escape(a) + r"(?=\s)", body))
        for a in sentences.EN_ABBREVIATIONS
    )
    assert occurrences > 0, "略語が 0 件 — 対照が成り立たない"
    assert without - with_rules == occurrences, (
        f"寄与 {without - with_rules} 対 出現 {occurrences}"
    )

    # 陰性対照 — 略語の直後で切れた文が無い。
    # `Mr. Samsa` を切ると「Samsa …」で始まる文ができる。
    sents = sentences.sentences_for(en)
    assert not [s for s in sents if s.text.startswith("Samsa")]
    assert any("Mr. Samsa" in s.text for s in sents), "対照の前提が崩れている"


@pytest.mark.validation
def test_t020_ja_quotative_contribution_matches_occurrences(editions):
    """T-020(G-15)— 日本語の引用助詞規則の寄与が `」と` / `」など` と一致する。"""
    ja = editions["ja_aozora49866"]
    with_rules = sentences.count_for(ja, rules=True)
    without = sentences.count_for(ja, rules=False)

    body = "\x00".join(p.text for p in ja.paragraphs)  # 段落をまたがせない
    occurrences = sum(
        len(re.findall("」" + q, body)) for q in sentences.JA_QUOTATIVE
    )
    assert occurrences > 0, "引用助詞が 0 件 — 対照が成り立たない"
    assert without - with_rules == occurrences, (
        f"寄与 {without - with_rules} 対 出現 {occurrences}"
    )


# --- T-021: 正規化の置換(G-16 / HC-174)-------------------------------------


@pytest.mark.validation
def test_t021_normalize_only_folds_whitespace():
    """T-021(G-16)— 正規化が畳んでよい空白の除去以外を行わない。

    とくに**全角空白 U+3000 が保存される**。青空文庫の本文は感嘆符・疑問符の後に
    全角アキを置く組版で、`re.sub(r"\\s+", " ")` はこれを黙って半角に潰す。
    1 対 1 の置換なので字数も件数も動かず、どの集計にも現れない(HC-174)。
    """
    raw = (ingest.RAW_DIR / "ja_aozora49866.html").read_bytes().decode("shift_jis")
    main = raw.split('<div class="main_text">', 1)[1]
    main = main.split('<div class="bibliographical_information">', 1)[0]
    fragments = [
        re.sub(r"<[^>]+>", "", f)
        for f in re.split(r"<br\s*/?>", main)
    ]
    fragments = [f for f in fragments if f.strip()]
    assert fragments, "走査対象が空"

    # **端と内部を分ける。** 段落先頭の字下げ(全角空白)は `strip` で落ちるのが
    # 正しい。禁じたいのは**内部**の全角空白が畳まれることである(HC-175)。
    ideographic_before = ideographic_after = 0
    stripped_indents = 0
    for f in fragments:
        after = ingest.normalize(f)
        if f.strip() != f:
            stripped_indents += 1
        diff = ingest.char_diff(f.strip(), after)
        # 増えてよいのは半角空白だけ(畳んだ結果)、減ってよいのは畳める空白だけ
        for ch, delta in diff.items():
            if delta > 0:
                assert ch == " ", f"意図しない文字が増えた: {ch!r} ({delta:+d})"
            else:
                assert ingest._FOLDABLE_SPACE.fullmatch(ch), (
                    f"意図しない文字が減った: {ch!r} ({delta:+d})"
                )
        ideographic_before += f.strip().count("　")
        ideographic_after += after.count("　")

    assert ideographic_before > 0, "内部の全角空白が 1 つも無い — 対照が成り立たない"
    assert stripped_indents > 0, "字下げのある段落が無い — 端と内部を分ける前提が崩れている"
    assert ideographic_after == ideographic_before, (
        f"内部の全角空白が {ideographic_before} → {ideographic_after} に減った"
    )


@pytest.mark.unit
def test_t021_positive_control_old_normalizer():
    """T-021 — 旧実装(`\\s+` で畳む)はこの検査に落ちる(陽性対照)。

    これが無いと、`char_diff` が常に空を返す実装でも上のケースは緑になる。
    """
    sample = "むりだ！　つぎの文。"
    old = re.sub(r"\s+", " ", sample).strip()
    diff = ingest.char_diff(sample, old)
    assert diff.get("　") == -1, "旧実装で全角空白が消えていない — 前提が崩れている"
    assert sample.count("　") == 1 and old.count("　") == 0
    # 長さは変わらない —— これが「数では捕まらない」ことの実証
    assert len(sample) == len(old)

    new = ingest.normalize(sample)
    assert new.count("　") == 1, "新実装が全角空白を保存していない"
