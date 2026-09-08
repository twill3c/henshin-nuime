"""自前和訳の落丁検査と訳語登録簿(T-068〜T-073)。

**検査に歯があることを、まず確かめる。** 訳文が全部そろっている状態で「OK」が出ても、
検査が何も見ていないだけかもしれない。だから各検査に**陽性対照**を置く ——
訳文を一箇所壊したら、その検査が、その箇所を指して落ちること。
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline import translate

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def src():
    return translate.load_source()


@pytest.fixture(scope="session")
def ja():
    return translate.load_translation()


@pytest.fixture(scope="session")
def glossary():
    return translate.load_glossary()


def _mutate(ja, index, old, new):
    return [dataclasses.replace(r, text=r.text.replace(old, new))
            if r.index == index and r.chapter == 1 else r for r in ja]


# --- T-068 段落の落丁 --------------------------------------------------------


def test_paragraphs_are_intact(src, ja):
    assert translate.check_paragraphs(src, ja) == []
    assert ja, "訳文が一段落も無い"


def test_paragraph_gap_is_caught(src, ja):
    """**陽性対照** — 途中を抜かすと落ちる。末尾で止めるのは落丁ではない。"""
    dropped = [r for r in ja if not (r.chapter == 1 and r.index == 2)]
    errs = translate.check_paragraphs(src, dropped)
    assert any("飛ばしている" in e for e in errs), errs
    # 末尾を切っただけなら落ちない(まだ訳していないだけ)
    head = [r for r in ja if r.chapter == 1 and r.index <= 5]
    assert translate.check_paragraphs(src, head) == []


def test_unknown_paragraph_is_caught(src, ja):
    ghost = dataclasses.replace(ja[0], index=999)
    errs = translate.check_paragraphs(src, [*ja, ghost])
    assert any("原文にその段落が無い" in e for e in errs), errs


# --- T-069 固有名の悉皆 ------------------------------------------------------


def test_names_are_present(src, ja, glossary):
    assert translate.check_names(src, ja, glossary) == []


def test_missing_name_is_caught(src, ja, glossary):
    """**陽性対照** — 1-1 から名前を消すと、その段落を指して落ちる。"""
    errs = translate.check_names(src, _mutate(ja, 1, "グレーゴル・ザムザ", "彼"),
                                 glossary)
    assert any("1-1" in e and "Gregor" in e for e in errs), errs
    assert any("Samsa" in e for e in errs), errs


def test_name_registry_matches_the_english_signal(src, glossary):
    """登録簿の名前が、独語側で実際に出る語であること。

    **独語の大文字は固有名の印にならない**(全名詞が大文字)。だから名前は
    英訳の文中大文字から拾った。ここでは拾った結果が独語原文に実在することを見る。
    """
    names = [k for k, v in glossary.items() if v.get("kind") == "name"]
    assert {"Gregor", "Samsa", "Grete"} <= set(names)
    for term in names:
        entry = glossary[term]
        assert any(translate.term_pattern(term, entry).search(s.text)
                   for s in src), term


# --- T-070 数の落丁 ----------------------------------------------------------


def test_numbers_are_present(src, ja):
    assert translate.check_numbers(src, ja) == []


def test_missing_number_is_caught(src, ja):
    """**陽性対照** — 時刻を消すと落ちる。"""
    errs = translate.check_numbers(src, _mutate(ja, 6, "六時半", "その時刻"))
    assert any("1-6" in e and "halb sieben" in e for e in errs), errs


def test_no_arabic_digits_in_the_source(src):
    """**照合を算用数字で組めない根拠**。原文に算用数字は 1 つも無い。"""
    assert not [s.pid for s in src if any(c.isdigit() for c in s.text)]


def test_indefinite_article_is_not_a_number(src):
    """`ein`/`eine` を数に数えない根拠 —— 200 件超あり、ほとんどが不定冠詞。

    数えていたら、正しい訳文が大量に落ちていた。
    """
    assert "ein" not in translate.NUMERALS
    assert "eine" not in translate.NUMERALS
    import re
    n = sum(len(re.findall(r"\bein(e|en|em|er)?\b", s.text)) for s in src)
    assert n > 150, n


def test_clock_idiom_is_matched_as_a_phrase(src):
    """**時刻は語ごとに訳せない。** 独語は次の時を言い、日本語はその時を言う。

    `halb sieben` = 六時半。`halb` と `sieben` を別々に数えると、正しい訳が落ちる。
    最長一致で言い回しとして拾い、その範囲を二重に数えないことを確かめる。
    """
    p = next(s for s in src if s.chapter == 1 and s.index == 6)
    found = [w for w, _ in translate.number_spans(p.text)]
    assert "halb sieben" in found
    # 「halb sieben」1 件を拾ったぶん、単独の halb / sieben はそのぶん減る
    assert found.count("sieben") == 1  # 「sieben Uhr」の 1 件だけ
    forms = dict(translate.number_spans(p.text))["halb sieben"]
    assert "六時半" in forms and "七時半" not in forms


def test_halb_is_split_into_quantity_and_compound(src):
    """**語頭の `halb-` は数量ではなく語構成要素**(`Halbschlaf` = まどろみ)。

    ここで大事なのは、例外を一つ足したのではなく**数え切った**ことである。
    本文の `halb` 系は 8 例しかなく、単独 5 / 合成 3 に割れる。
    数え切ってあるので言い回しの表はこれ以上増えない ——
    **開いた例外表と、閉じた列挙は別物**である。
    """
    import re
    total = sum(len(re.findall(r"\b[Hh]alb\w*", s.text)) for s in src)
    assert total == 8, total
    picked: dict[str, int] = {}
    for s in src:
        for w, _ in translate.number_spans(s.text):
            if "alb" in w:
                picked[w] = picked.get(w, 0) + 1
    assert sum(picked.values()) == total
    assert picked["halb"] == 4          # 単独 5 のうち halb sieben を除いた 4
    assert picked["halb sieben"] == 1
    assert {"Halbschlaf", "halbverfault", "halblaut"} <= picked.keys()
    # 合成語は「半」を要求しない形で登録されている
    assert "まどろみ" in translate.IDIOMS["Halbschlaf"]


def test_stem_collisions_are_excluded(src):
    """`achten`(注意する)は八ではない。`zweifeln`(疑う)は二ではない。

    **これは緑にするための例外ではない** —— 語そのものが数ではないことを、
    原文に実在することで示す。
    """
    import re
    joined = " ".join(s.text for s in src)
    assert re.search(r"\bachte(n|te)\b", joined)
    assert re.search(r"\bzweifel", joined, re.IGNORECASE)
    for s in src:
        for word, _ in translate.number_spans(s.text):
            assert not word.startswith(translate.NOT_NUMBERS)


# --- T-074 字数比(三本の検査の穴)-------------------------------------------


def test_lengths_are_within_the_band(src, ja):
    assert translate.check_lengths(src, ja) == []


def test_a_half_dropped_paragraph_is_caught(src, ja, glossary):
    """**三本の落丁検査には穴がある。**

    固有名も数も含まない文を落とすと、段落は残り、名前も数も揃うので、
    どの検査も黙る。ここではその穴が実在することと、字数比が塞ぐことの
    両方を、同じ壊し方で示す。
    """
    cut = [dataclasses.replace(r, text=r.text[: len(r.text) // 2])
           if r.chapter == 1 and r.index == 8 else r for r in ja]
    # 穴が実在する —— 他の三本は黙る
    assert translate.check_names(src, cut, glossary) == []
    assert translate.check_numbers(src, cut) == []
    assert translate.check_paragraphs(src, cut) == []
    # 字数比だけが捕まえる
    errs = translate.check_lengths(src, cut)
    assert any("1-8" in e for e in errs), errs


def test_length_band_does_not_fire_on_style(src, ja):
    """帯は**広く**取ってある。実測の幅(0.37〜0.48)より広い。"""
    by_key = {(s.chapter, s.index): s for s in src}
    ratios = [len(r.text) / len(by_key[(r.chapter, r.index)].text) for r in ja]
    assert 0.3 < min(ratios) and max(ratios) < 0.6, (min(ratios), max(ratios))
    lo, hi = translate.LENGTH_BAND
    assert lo < 0.7 and hi > 1.4


# --- T-071 訳語登録簿 --------------------------------------------------------


def test_glossary_is_consistent(src, ja, glossary):
    assert translate.check_glossary(src, ja, glossary) == []


def test_wobbling_a_term_is_caught(src, ja, glossary):
    """**陽性対照** — 一段落だけ別の訳語にすると落ちる。"""
    errs = translate.check_glossary(src, _mutate(ja, 5, "社長", "ボス"), glossary)
    assert any("1-5" in e and "Chef" in e for e in errs), errs


def test_registry_numbers_are_measured(src, glossary):
    """登録簿に書いた件数・初出が原文と一致する(T-071 / L11 で実際に踏んだ)。"""
    assert translate.check_registry_facts(src, glossary) == []


def test_registry_lies_are_caught(src, glossary):
    """**陽性対照** — もっともらしい数字を書いたら落ちる。"""
    faked = {k: {**v} for k, v in glossary.items()}
    faked["Gregor"]["count"] = 999
    faked["Samsa"]["first"] = "3-1"
    errs = translate.check_registry_facts(src, faked)
    assert any("Gregor" in e and "件数" in e for e in errs), errs
    assert any("Samsa" in e and "初出" in e for e in errs), errs


def test_prefix_match_would_swallow_another_word(src, glossary):
    """`Anna` を前方一致にすると `Annahme` を拾う —— 実測で決めた線引き。"""
    assert glossary["Anna"]["match"] == "word"
    loose = {"Anna": {**glossary["Anna"], "match": "prefix"}}
    hit = [s.pid for s in src
           if translate.term_pattern("Anna", loose["Anna"]).search(s.text)]
    strict = [s.pid for s in src
              if translate.term_pattern("Anna", glossary["Anna"]).search(s.text)]
    assert len(hit) > len(strict)


# --- T-072 字種検査 ----------------------------------------------------------


def test_translation_is_scanned_for_stray_scripts():
    """**`data/` は字種検査の既定の走査から外れている。**

    外部データの原文を持つ場所なので除外されているが、そこに自前の文章を置くと
    **無検査のまま出荷される**。だから経路を明示して走らせる。
    """
    targets = sorted(translate.TRANSLATION_DIR.glob("*.json"))
    assert targets, "訳文が一つも無い"
    r = subprocess.run(
        [sys.executable, "harness/text_hygiene.py", *[str(p) for p in targets]],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "違反 0 件" in r.stdout


def test_the_scanner_would_catch_a_stray_script(tmp_path):
    """**陽性対照** — 同じ検査器が、異種文字を混ぜた同じ形のファイルで落ちる。"""
    bad = tmp_path / "ja_own_test.json"
    bad.write_text(json.dumps({"text": "グレーゴルは독일어で考えた"},
                              ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, "harness/text_hygiene.py", str(bad)],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode != 0, r.stdout


# --- T-073 循環の禁止 --------------------------------------------------------


def test_translator_never_reads_the_japanese_edition():
    """自前訳の器が原田訳を読まない(構造の歯止め)。

    読めば「訳者差」を測る対照が自分自身の写しになる。
    **これは書く人が原田訳を読まないことの保証ではない** —— そこは手続きの約束。
    """
    text = (ROOT / "pipeline" / "translate.py").read_text(encoding="utf-8")
    code = text.split('"""', 2)[2]  # 冒頭の説明文は除いて、コードだけを見る
    assert "ja_aozora" not in code
    assert "en_pg5200" not in code
    assert translate.SOURCE.name == "de_pg22367.json"


def test_cli_runs(capsys):
    """起動経路(T-015 と同じ主旨)。"""
    assert translate.main([]) == 0
    assert "自前和訳" in capsys.readouterr().out
