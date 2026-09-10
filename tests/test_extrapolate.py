"""外挿検証の取り込みと前提(T-082〜T-085)。

**権利の前提を検査に置く。** L15 で、英訳が公有だという想定が崩れた(HC-255)。
配布元の版が差し替われば前提も変わりうるので、
「COPYRIGHTED と書いてあること」自体を検査で固定する ——
消えたら、配らない決まりを見直す前に条項を読み直せ、という合図になる。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import extrapolate as X

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not (RAW / "de_pg69327.txt").exists(),
    reason="外挿セットの原本が手元に無い(README の入手手順を見ること)",
)


# --- T-082 権利の前提 --------------------------------------------------------


def test_english_translation_is_still_copyrighted():
    """**英訳は公有ではない。** この表記が消えたら前提が変わったということ。"""
    raw = (RAW / "en_pg7849.txt").read_text(encoding="utf-8")
    assert "COPYRIGHTED" in raw
    assert "Translation Copyright" in raw
    # 許諾収録の作品には 1.E.3 が及ぶ(1.E.1〜1.E.7 + 権利者の追加条件)
    assert "posted with the permission of the copyright holder" in raw


def test_german_and_japanese_are_public_domain():
    """独語(PG)と日本語(青空)には著作権存続の表記が無い。"""
    de = (RAW / "de_pg69327.txt").read_text(encoding="utf-8")
    assert "COPYRIGHTED" not in de
    assert "Verlag die Schmiede, 1925" in de
    ja = (RAW / "ja_aozora49863.html").read_bytes().decode("shift_jis")
    # 訳者と底本の記載が残っていること(青空の取り扱い規準)
    assert "原田義人" in ja or "原田　義人" in ja
    assert "新潮文庫" in ja


def test_the_copyrighted_original_is_not_in_the_repository():
    """**公有でない原本を版に入れない。**

    手元に置くのは計測のためで、版に入れれば再配布になる。
    `.gitignore` に経路が書いてあることを検査で固定する ——
    次に触る人が `git add -A` で巻き込まないように。
    """
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/en_pg7849.txt" in ignore
    # 公有の原本のほうは版に残す(取り違えていないことの確認)
    assert "data/raw/de_pg69327.txt" not in ignore
    assert "data/raw/ja_aozora49863.html" not in ignore


def test_loader_refuses_if_the_copyright_notice_disappears(monkeypatch, tmp_path):
    """**陽性対照** —— 表記が消えた原本を渡すと、読み込み器が止まる。"""
    fake = tmp_path / "raw"
    fake.mkdir()
    (fake / "en_pg7849.txt").write_text("no notice here", encoding="utf-8")
    monkeypatch.setattr(X, "RAW", fake)
    with pytest.raises(X.ExtrapolationError, match="COPYRIGHTED"):
        X.load_en()


# --- T-083 章の切り方 --------------------------------------------------------


def test_ten_chapters_in_all_three():
    for load in (X.load_de, X.load_en, X.load_ja):
        paras = load()
        assert {p.chapter for p in paras} == set(range(1, 11))


def test_chapter_titles_and_afterword_are_not_body():
    """**本文でないものを本文として数えない。**

    (1) 各章の第 1 段落は章題、(2) 独語版には Max Brod の後書きが続く。
    どちらも『変身』には無かったので扱いを決めておらず、一度数え込んだ。
    """
    de = X.load_de()
    # 章題(独語は全部大文字の短い行)が本文に残っていない
    firsts = [next(p.text for p in de if p.chapter == c) for c in range(1, 11)]
    assert not any(t.isupper() and len(t) < 80 for t in firsts), firsts
    # 後書きが混ざっていない
    assert not any("NACHWORT" in p.text for p in de)
    # 英訳も章題を落としている
    en = X.load_en()
    assert not any(p.text.startswith("Arrest--Conversation") for p in en)


# --- T-084 段落構造は一致しない(『変身』との対比)---------------------------


def test_paragraph_structures_do_not_match():
    """**背骨のオラクルが別の本では成り立たない。**

    『変身』では独英が 97 対 97 で総数も章別も完全一致し、段落一致率という
    物差しはその上に立っていた。『審判』では独 140 対 英 126 で、
    その物差しがそもそも定義できない。**この不一致を検査で固定する** ——
    もし将来これが一致するようになったら、取り込みか原本が変わったということ。
    """
    de, en = X.load_de(), X.load_en()
    assert len(de) == 140
    assert len(en) == 126
    assert len(de) != len(en)
    # 併合であって分割ではない(英のほうが少ない)ことを、章ごとにも確かめる
    for c in range(1, 11):
        d = sum(1 for p in de if p.chapter == c)
        e = sum(1 for p in en if p.chapter == c)
        assert d >= e, (c, d, e)


def test_japanese_is_much_finer_than_in_die_verwandlung():
    """日本語の細分の度合いも作品で違う(1.70 倍 → 10.7 倍)。"""
    de, ja = X.load_de(), X.load_ja()
    ratio = len(ja) / len(de)
    assert ratio > 5, ratio


# --- T-085 規則を当てはめ直さない --------------------------------------------


def test_sentence_rules_are_reused_not_refitted():
    """文分割は『変身』で決めた規則をそのまま呼ぶ。

    作品に合わせて調整すると、外挿が外挿でなくなる。
    ここでは「自前の分割器を持っていないこと」を構造で確かめる。
    """
    import re as _re

    src = (ROOT / "pipeline" / "extrapolate.py").read_text(encoding="utf-8")
    assert "sentences.spans" in src

    # **見るのは正規表現だけ。** 最初これを「本文に `。` が出てこないこと」で
    # 書いたら、日本語の**コメント**の句点を分割規則と読んで落ちた ——
    # 述語が広すぎると、正しい実装を撃つ(HC-074)。
    patterns = _re.findall(r"re\.(?:compile|split|finditer|search|match)\(\s*(r?[\"'][^\"']*[\"'])",
                           src)
    assert patterns, "正規表現が一つも拾えていない — 走査が働いていない"
    for p in patterns:
        assert "!?" not in p, p          # 文末の候補を並べた形
        assert "。" not in p, p
        assert "」" not in p, p


def test_structure_counts_are_stable():
    st = X.structure()
    assert st["de"]["paragraphs"] == 140
    assert st["en"]["paragraphs"] == 126
    assert st["ja"]["paragraphs"] == 1503
    assert sum(st["de"]["chapter_paragraphs"]) == st["de"]["paragraphs"]
