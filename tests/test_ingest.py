"""取り込み器の検査。TEST_SPEC の T-001〜T-010 に対応する。

期待値の出所は原則「二つの独立したファイルが互いに一致する」という不変量であり、
件数の直書きは避ける(HC-016)。定数を置いてよいのは T-013(SPEC の実測表との突合)だけ。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import ingest  # noqa: E402
from harness import text_hygiene  # noqa: E402


@pytest.fixture(scope="module")
def editions():
    return ingest.load_all()


# --- T-001 / T-002: PG の包装 -------------------------------------------------


@pytest.mark.unit
def test_t001_no_pg_wrapper_left_in_paragraphs(editions):
    """T-001 — PG のヘッダ・フッタ・許諾文が段落本文に残らない。"""
    forbidden = [
        "PROJECT GUTENBERG",
        "Project Gutenberg",
        "gutenberg.org",
        "Distributed Proofreading",
    ]
    for eid, ed in editions.items():
        for p in ed.paragraphs:
            for f in forbidden:
                assert f not in p.text, f"{p.pid} に PG の包装が残っている: {f!r}"


@pytest.mark.unit
def test_t002_pg_wrapper_positive_control():
    """T-002 — 標識を欠く入力は例外で落ちる(陽性対照)。

    「除去できた」は「標識を探した」を意味しない。標識が無いときに黙って
    全文を返す実装でも T-001 は緑になるので、ここで撃たせる。
    """
    with pytest.raises(ingest.IngestError):
        ingest.strip_pg_wrapper("本文だけで標識が無い")
    # 順序が逆(END が先)も落ちること
    reversed_doc = (
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
    )
    with pytest.raises(ingest.IngestError):
        ingest.strip_pg_wrapper(reversed_doc)
    # 陰性対照 — 正しい包装は落ちず、中身だけを返す
    ok = (
        "front matter\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "BODY\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "back matter\n"
    )
    body = ingest.strip_pg_wrapper(ok)
    assert body.strip() == "BODY"
    assert "front matter" not in body and "back matter" not in body


# --- T-003: 章 ---------------------------------------------------------------


@pytest.mark.unit
def test_t003_chapter_count_and_order(editions):
    """T-003 — 三版とも章は 3 つ。壊した並びは例外で落ちる(陽性対照)。"""
    for eid, ed in editions.items():
        chapters = sorted({p.chapter for p in ed.paragraphs})
        assert chapters == [1, 2, 3], f"{eid} の章が {chapters}"

    # 陽性対照: 章標識を 1 つ削ると落ちる
    body = "\nI.\n\na\n\nII.\n\nb\n\nIII.\n\nc\n"
    assert len(ingest.split_pg_chapters(body)) == 3
    with pytest.raises(ingest.IngestError):
        ingest.split_pg_chapters(body.replace("\nII.\n", "\n"))
    # 陽性対照: 並びが I/II/III でなければ落ちる
    with pytest.raises(ingest.IngestError):
        ingest.split_pg_chapters("\nI.\n\na\n\nIII.\n\nb\n\nII.\n\nc\n")


# --- T-004: 独英の一致 -------------------------------------------------------


@pytest.mark.integration
def test_t004_de_en_paragraph_structure_matches(editions):
    """T-004 — 独と英の段落構造が一致する。

    これは本企画の非循環オラクルの中核である。二つのファイルは互いを参照せずに
    作られており、片方から他方を導いていない。定数ではなく相互一致で書く。
    """
    de = ingest.chapter_counts(editions["de_pg22367"].paragraphs)
    en = ingest.chapter_counts(editions["en_pg5200"].paragraphs)
    assert de == en, f"章別段落数が食い違う 独={de} 英={en}"
    assert sum(de) == sum(en)
    # 走査対象が空でないこと
    assert sum(de) > 0


# --- T-005: 細分性の必要条件 --------------------------------------------------


@pytest.mark.integration
def test_t005_ja_refines_de(editions):
    """T-005(G-07 の必要条件)— 各章で 日本語の段落数 ≥ 独語の段落数。

    日本語は会話文が行立てになるため独より細かい。細分であるための必要条件が
    章ごとに成り立つことをここで押さえる。写像そのものの検査は L2(G-07)。
    """
    de = ingest.chapter_counts(editions["de_pg22367"].paragraphs)
    ja = ingest.chapter_counts(editions["ja_aozora49866"].paragraphs)
    assert len(de) == len(ja) == 3
    for i, (d, j) in enumerate(zip(de, ja), start=1):
        assert j >= d, f"第{i}章: 日本語 {j} < 独語 {d} — 細分ではありえない"


# --- T-006: 引用符の均衡 ------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "eid,opener,closer",
    [
        ("de_pg22367", "»", "«"),
        ("en_pg5200", "“", "”"),
        ("ja_aozora49866", "「", "」"),
    ],
)
def test_t006_quote_balance(editions, eid, opener, closer):
    """T-006 — 各版で開き括弧と閉じ括弧の数が等しい。"""
    text = " ".join(p.text for p in editions[eid].paragraphs)
    n_open, n_close = text.count(opener), text.count(closer)
    assert n_open > 0, f"{eid} に開き括弧が 1 つも無い — 走査対象が空"
    assert n_open == n_close, f"{eid} 開き {n_open} 閉じ {n_close}"


# --- T-007: 段落テキストの衛生 ------------------------------------------------


@pytest.mark.unit
def test_t007_paragraph_text_hygiene(editions):
    """T-007 — 空文字・前後空白・改行を含まない。"""
    for eid, ed in editions.items():
        assert ed.paragraphs, f"{eid} の段落が 0 件"
        for p in ed.paragraphs:
            assert p.text, f"{p.pid} が空"
            assert p.text == p.text.strip(), f"{p.pid} の前後に空白"
            assert "\n" not in p.text and "\r" not in p.text, f"{p.pid} に改行"
            assert "  " not in p.text, f"{p.pid} に連続空白"


# --- T-008: ルビ --------------------------------------------------------------


@pytest.mark.integration
def test_t008_ruby_base_kept_reading_dropped():
    """T-008 — ルビの親文字は本文に残り、振仮名は本文に混入していない。

    振仮名は本文ではなく注記なので、段落テキストからは外れているのが正しい。

    **字面では検査できない。** 「親文字 + 振仮名」の並びは日本語の地の文に
    偶然現れる(「玄関の間まで」は 間 + ま に見える)。そこで判定は
    **二つの独立した除去経路の一致**で行う:

      経路A  ruby 要素を親文字へ書き換えてからタグを剥ぐ(実装が使う規則)
      経路B  rt・rp 要素を丸ごと落としてからタグを剥ぐ(振仮名を持たない別経路)

    どちらも「振仮名が消え親文字が残る」を別の書き方で表す。一致すれば、
    片方の規則がたまたま働いていない状態を捕まえられる。
    """
    raw = (ingest.RAW_DIR / "ja_aozora49866.html").read_bytes().decode("shift_jis")
    pairs = ingest.extract_aozora_ruby(raw)
    assert pairs, "ルビが 1 件も見つからない — 走査対象が空"

    main = raw.split('<div class="main_text">', 1)[1]
    main = main.split('<div class="bibliographical_information">', 1)[0]
    fragments = [f for f in re.split(r"<br\s*/?>", main) if "<ruby>" in f]
    assert fragments, "ルビを含む断片が無い — 対照の前提が崩れている"

    def path_b(fragment: str) -> str:
        without_reading = re.sub(r"<r[tp]>.*?</r[tp]>", "", fragment, flags=re.S)
        return ingest.normalize(re.sub(r"<[^>]+>", "", without_reading))

    def naive(fragment: str) -> str:
        """陽性対照 — 素通しでタグだけ剥ぐと振仮名が本文に残る。"""
        return ingest.normalize(re.sub(r"<[^>]+>", "", fragment))

    leaked = 0
    for f in fragments:
        assert ingest.strip_aozora_markup(f) == path_b(f), "二経路が食い違う"
        if naive(f) != path_b(f):
            leaked += 1
    assert leaked == len(fragments), (
        "素通しの剥ぎ取りと結果が同じ断片がある — この対照は何も検出していない"
    )

    # 親文字は本文に残っていること
    body = " ".join(p.text for p in ingest.load_ja().paragraphs)
    for base, _reading in pairs:
        assert base in body, f"親文字 {base!r} が本文から消えた"


# --- T-009: 決定論 ------------------------------------------------------------


@pytest.mark.unit
def test_t009_ingest_is_deterministic():
    """T-009 — 同じ生ファイルから二度組み立てて完全一致する。"""
    a, b = ingest.load_all(), ingest.load_all()
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].paragraphs == b[k].paragraphs


# --- T-010: 字種衛生 ----------------------------------------------------------


@pytest.mark.validation
def test_t010_char_hygiene_on_bodies(editions):
    """T-010(G-02)— 本文に異種文字・制御文字が混入していない。

    検査器自身の陽性対照を先に通す(HC-080)。「違反 0 件」は検査器が
    死んでいても同じ値を返すため、道具の生存確認なしにこの緑は読めない。
    """
    assert text_hygiene.self_test() == [], "字種検査器の自己対照が落ちている"

    for eid, ed in editions.items():
        for p in ed.paragraphs:
            bad = text_hygiene.scan_line(p.text)
            assert not bad, f"{p.pid} に {bad}"

    # 陰性対照は**実データの正常な部分から取る**(HC-074)。
    # 独語のウムラウト・ß と英語のカーリークォートを撃たないこと。
    de_text = " ".join(p.text for p in editions["de_pg22367"].paragraphs)
    umlaut_para = next(p for p in editions["de_pg22367"].paragraphs
                       if "ü" in p.text and "ß" in p.text)
    assert text_hygiene.scan_line(umlaut_para.text) == []
    assert "ü" in de_text and "ß" in de_text  # 対照が成り立つ前提の固定
    en_para = next(p for p in editions["en_pg5200"].paragraphs
                   if "“" in p.text and "’" in p.text)
    assert text_hygiene.scan_line(en_para.text) == []
