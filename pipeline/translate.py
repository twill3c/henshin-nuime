"""自前の和訳と、その落丁検査・訳語登録簿(F-06 / G-12 / G-13)。

**この訳は独語原文からだけ作る。** 原田訳を見て作れば、「訳者差」を測るための対照が
自分自身の写しになる。ここには**構造の歯止め**を置いてある —— このモジュールは
日本語版(`ja_aozora49866`)を一度も読まない。読む口が無い(T-073)。

**歯止めの限界も書いておく。** ファイルを読まないことは、書く人が原田訳を読まないことの
保証ではない。そこは手続きの約束であって検査では固定できない(SPEC §8)。
検査で固定できるのは「このコードが読まない」ことだけである。

落丁検査は三つに分けてある。**一つの数字にまとめない** —— どこが落ちたかで直し方が違う。

  段落     訳した段落が原文に実在し、重複せず、章の途中を飛ばしていない(T-068)
  固有名   原文の段落に出る登録済みの名前が、訳文にも登録の訳語で出る(T-069)
  数       原文の段落に出る数が訳文にも出る(T-070)

**数の照合を算用数字で組んではいけない。** 独語本文に算用数字は **0 件**で、
数はすべて綴りで書かれる(`sieben Uhr`)。さらに `ein`/`eine` は 203 件あるが
ほとんどが不定冠詞なので、数として数えれば偽陽性だらけになる。だから `zwei` 以上と
`halb`・`Uhr` だけを見る。**この線引きは実測から引いた**(2026-09-08)。

**固有名は独語側からは拾えない。** 独語は全ての名詞を大文字にするので、大文字は
固有名の印にならない。英訳の文中大文字を印にして登録簿を作った(Gregor 229 /
Samsa 34 / Grete 21)。登録簿は人が確かめて `data/translation/glossary.json` に置く。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "editions" / "de_pg22367.json"
TRANSLATION_DIR = ROOT / "data" / "translation"
GLOSSARY = TRANSLATION_DIR / "glossary.json"

# 数として数える綴り。**`ein`/`eine` は入れない**(203 件のほとんどが不定冠詞)。
# 見出し語 → 訳文側で認める形。訳文が漢数字でも算用数字でも通す。
#
# **時刻の言い方は語ごとに訳せない。** 独語は**次の**時を基準に言い、日本語は**その**時を
# 基準に言う —— `halb sieben` は「七時半」ではなく **六時半**、`dreiviertel sieben` は
# 「七時十五分前」である。だから `halb` と `sieben` を別々に数えると、正しい訳文が
# 落ちる。照合の単位は語ではなく**言い回し**でなければならない。
# 長いものから先に当て、当たった範囲は短いほうに二重に数えさせない(最長一致)。
# 語ではなく**言い回し**で照合するもの。二種類ある。
#
#   時刻      独語は次の時を基準に言う(`halb sieben` = 六時半)
#   合成語    数を表す語が語構成要素になっていて、日本語がその数を出さない
#             (`Halbschlaf` = まどろみ)
#
# **これは開いた例外表ではない。** `halb*` は本文に 8 例しかなく、全部数えた ——
# 単独 5 例(halb sieben / halb vorüber / halb unbewußt / halb fragend / halb erhob)は
# 数量なのでそのまま照合し、合成語 3 例だけをここに置く。増えようがない(2026-09-08 実測)。
IDIOMS: dict[str, tuple[str, ...]] = {
    "halb sieben": ("六時半", "六時三十分", "六時 30 分"),
    "einviertel acht": ("七時十五分", "七時 15 分"),
    "dreiviertel": ("四十五分", "十五分前", "45 分"),
    "viertelstund": ("十五分", "四半時", "15 分"),
    "viertelstünd": ("十五分", "四半時", "15 分"),
    "Halbschlaf": ("まどろみ", "うたた寝", "半"),
    "halbverfault": ("半ば腐", "腐りかけ", "半"),
    "halblaut": ("小声", "半ば声", "低い声", "半"),
}

# **綴りが数と同じでも数でない語がある。** 数える前に外す。
# これは「緑にするための例外」ではない —— 語そのものが数ではない(実測 2026-09-08):
#   achten / achtete   6 件  「注意する・気を配る」であって八ではない
#   zweifel…           3 件  「疑い・疑った」であって二ではない
NOT_NUMBERS = ("achten", "achtete", "zweifel")

NUMERALS: dict[str, tuple[str, ...]] = {
    "zwei": ("二", "2", "ふた", "両"),
    "drei": ("三", "3", "み"),
    "vier": ("四", "4", "よ"),
    "fünf": ("五", "5", "いつ"),
    "sechs": ("六", "6", "む"),
    "sieben": ("七", "7", "なな"),
    "acht": ("八", "8", "や"),
    "neun": ("九", "9"),
    "zehn": ("十", "10"),
    # **十代の数は語幹が変わる。** `sechzehn`/`siebzehn` に `sechs`/`sieben` は
    # 含まれないので、語幹の前方一致では拾えない(実測 各 1 例・2026-09-08)。
    # `vierzehn` は `vier` で拾えている —— 拾えるものと拾えないものが混ざるので、
    # 「前方一致にしたから十代も見ている」とは言えない。
    "sechzehn": ("十六", "16"),
    "siebzehn": ("十七", "17"),
    "elf": ("十一", "11"),
    "zwölf": ("十二", "12"),
    "hundert": ("百", "100"),
    "tausend": ("千", "1000"),
    "halb": ("半", "半分", "なかば"),
    "Uhr": ("時", "時計"),
}


class TranslationError(RuntimeError):
    """訳文と原文の対応が仮定と食い違ったときに投げる。"""


def term_pattern(term: str, entry: dict) -> re.Pattern:
    """原文で語を探す形。**語尾変化を拾うか、拾わないかを語ごとに決める。**

    既定は前方一致。独語は格変化するので `Prokurist` は `Prokuristen` 15 件を、
    `Gregor` は `Gregors` 45 件を取りこぼす(実測)。
    ただし前方一致は**別語を巻き込む** —— `Anna`(2 件)は `Annahme`(1 件)を拾う。
    だから登録簿の側に `"match": "word"` を書けるようにしてある。
    **どちらが正しいかは語ごとに実測して決める。**
    """
    esc = re.escape(term)
    return re.compile(rf"\b{esc}\b" if entry.get("match") == "word"
                      else rf"\b{esc}\w*")


@dataclass(frozen=True)
class Source:
    pid: str
    chapter: int
    index: int
    text: str


@dataclass(frozen=True)
class Rendered:
    """訳した段落一つ。`index` は独語原文の段落番号(章内の通番)。"""

    chapter: int
    index: int
    text: str
    note: str = ""


def load_source() -> list[Source]:
    """**独語原文だけ**を読む。日本語版を読む口はこのモジュールに無い。"""
    d = json.loads(SOURCE.read_text(encoding="utf-8"))
    return [Source(p["pid"], p["chapter"], p["index"], p["text"])
            for p in d["paragraphs"]]


def load_glossary() -> dict[str, dict]:
    if not GLOSSARY.exists():
        return {}
    return json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]


def load_translation() -> list[Rendered]:
    """章ごとのファイルを読む。無い章は単に無い(充填率は分子と分母で出す)。"""
    out: list[Rendered] = []
    if not TRANSLATION_DIR.exists():
        return out
    for path in sorted(TRANSLATION_DIR.glob("ja_own_ch*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        ch = d["chapter"]
        for p in d["paragraphs"]:
            out.append(Rendered(ch, p["index"], p["text"], p.get("note", "")))
    return out


# --- 落丁検査 ---------------------------------------------------------------


def check_paragraphs(src: list[Source], ja: list[Rendered]) -> list[str]:
    """段落の落丁(T-068)。**飛ばして先へ進んでいないこと**まで見る。"""
    errs: list[str] = []
    known = {(s.chapter, s.index) for s in src}
    seen: set[tuple[int, int]] = set()
    for r in ja:
        key = (r.chapter, r.index)
        if key not in known:
            errs.append(f"{r.chapter}-{r.index}: 原文にその段落が無い")
        if key in seen:
            errs.append(f"{r.chapter}-{r.index}: 二度訳されている")
        seen.add(key)
        if not r.text.strip():
            errs.append(f"{r.chapter}-{r.index}: 訳文が空")
    for ch in sorted({c for c, _ in seen}):
        idx = sorted(i for c, i in seen if c == ch)
        if idx != list(range(1, len(idx) + 1)):
            missing = sorted(set(range(1, max(idx) + 1)) - set(idx))
            errs.append(f"{ch} 章: 途中を飛ばしている(欠 {missing})")
    return errs


def _found(text: str, forms) -> bool:
    return any(f in text for f in forms)


def check_names(src: list[Source], ja: list[Rendered],
                glossary: dict[str, dict]) -> list[str]:
    """固有名の悉皆(T-069)。原文の段落に出る名前は訳文にも出る。"""
    names = {k: v for k, v in glossary.items() if v.get("kind") == "name"}
    by_key = {(s.chapter, s.index): s for s in src}
    errs: list[str] = []
    for r in ja:
        s = by_key.get((r.chapter, r.index))
        if s is None:
            continue
        for term, entry in names.items():
            if not term_pattern(term, entry).search(s.text):
                continue
            forms = [entry["ja"], *entry.get("also", [])]
            if not _found(r.text, forms):
                errs.append(f"{r.chapter}-{r.index}: 原文の {term} が訳文に無い"
                            f"(認める形: {'/'.join(forms)})")
    return errs


def number_spans(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """原文から数の言い回しを拾う。**最長一致** —— `halb sieben` を拾ったら、
    その範囲の `halb` と `sieben` は二重に数えない。
    """
    entries = sorted({**IDIOMS, **NUMERALS}.items(), key=lambda kv: -len(kv[0]))
    taken: list[tuple[int, int]] = []
    found: list[tuple[str, tuple[str, ...]]] = []
    for word, forms in entries:
        for m in re.finditer(rf"\b{re.escape(word)}\w*", text, re.IGNORECASE):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            if m.group().lower().startswith(NOT_NUMBERS):
                continue
            taken.append((m.start(), m.end()))
            found.append((word, forms))
    return found


def check_numbers(src: list[Source], ja: list[Rendered]) -> list[str]:
    """数の落丁(T-070)。綴りの数で見る —— 原文に算用数字は 0 件。"""
    by_key = {(s.chapter, s.index): s for s in src}
    errs: list[str] = []
    for r in ja:
        s = by_key.get((r.chapter, r.index))
        if s is None:
            continue
        for word, forms in number_spans(s.text):
            if not _found(r.text, forms):
                errs.append(f"{r.chapter}-{r.index}: 原文の {word} が訳文に無い"
                            f"(認める形: {'/'.join(forms)})")
    return errs


def check_glossary(src: list[Source], ja: list[Rendered],
                   glossary: dict[str, dict]) -> list[str]:
    """訳語の揺れ(T-071)。**同じ独語の語を二通りに訳していないこと。**

    登録簿にある語が出る段落では、訳文が登録済みの形のいずれかを使っていなければ
    落とす。登録簿を増やせば通るが、**増やすときは初出段落と理由を書く**決まりにしてある。
    """
    by_key = {(s.chapter, s.index): s for s in src}
    errs: list[str] = []
    for r in ja:
        s = by_key.get((r.chapter, r.index))
        if s is None:
            continue
        for term, entry in glossary.items():
            if entry.get("kind") == "name":
                continue  # 名前は check_names が悉皆で見る
            if not term_pattern(term, entry).search(s.text):
                continue
            forms = [entry["ja"], *entry.get("also", [])]
            if not _found(r.text, forms):
                errs.append(f"{r.chapter}-{r.index}: {term} の訳語が登録簿に無い形"
                            f"(登録: {'/'.join(forms)})")
    return errs


# 実測(第一章 30 段落・2026-09-08): 字数比 0.37〜0.48、中央値 0.408。
# 帯はこの実測から取るが、**文体の幅で落とさないよう広めに**取る。
# 半分落とせば 0.2 前後になるので、それは捕まえられる。
LENGTH_BAND = (0.6, 1.6)  # 中央値に対する倍率


def check_lengths(src: list[Source], ja: list[Rendered]) -> list[str]:
    """段落の中で文が落ちていないか(T-074)。

    **三本の落丁検査には穴がある。** 固有名も数も含まない文を一つ落とすと、
    段落は残り、名前も数も揃っているので、どれも黙る。字数の比はそれを捕まえる ——
    ただし**当てにしすぎない**。訳文の長さは文体でも動くので、帯は広く取り、
    外れたものを「落丁だ」ではなく「見ろ」として出す。
    """
    if len(ja) < 5:
        return []  # 中央値が定まらない
    by_key = {(s.chapter, s.index): s for s in src}
    ratios = []
    for r in ja:
        s = by_key.get((r.chapter, r.index))
        if s and s.text:
            ratios.append((len(r.text) / len(s.text), r))
    if not ratios:
        return []
    mid = sorted(x for x, _ in ratios)[len(ratios) // 2]
    lo, hi = mid * LENGTH_BAND[0], mid * LENGTH_BAND[1]
    return [f"{r.chapter}-{r.index}: 字数比 {x:.2f} が帯 {lo:.2f}〜{hi:.2f} の外"
            f"(中央値 {mid:.2f})" for x, r in ratios if not lo <= x <= hi]


def check_registry_facts(src: list[Source],
                         glossary: dict[str, dict]) -> list[str]:
    """**登録簿に書いた数字を、原文と突き合わせる。**

    L11 で実際に踏んだ形 —— 件数は実測から写したのに、初出段落は記憶で書き、
    13 語中 9 語が食い違った。もっともらしい数字は検査を素通りする。
    人が書かない約束にするのではなく、**書いたら落ちる形**にしておく。
    """
    errs: list[str] = []
    for term, entry in glossary.items():
        pat = term_pattern(term, entry)
        n = 0
        first: str | None = None
        for s in src:
            hits = pat.findall(s.text)
            if hits and first is None:
                first = f"{s.chapter}-{s.index}"
            n += len(hits)
        if n == 0:
            errs.append(f"{term}: 原文に一度も出ない")
        if entry.get("count") != n:
            errs.append(f"{term}: 件数 登録 {entry.get('count')} 対 実測 {n}")
        if entry.get("first") != first:
            errs.append(f"{term}: 初出 登録 {entry.get('first')} 対 実測 {first}")
    return errs


def registry_coverage(src: list[Source], ja: list[Rendered],
                      glossary: dict[str, dict]) -> dict[str, object]:
    """**登録簿がどれだけ届いているかを数える**(HC-237)。

    訳語の一貫性ゲートは、登録簿にある語しか見ない。だから「検査がある」ことと
    「検査が届いている」ことは別で、後者は被覆でしか言えない ——
    そして**被覆の穴はゲートの緑として現れる**ので、結果を見ていても気づけない。
    実際 L11 は `Reisender` を登録しておらず、訳語が揺れたまま緑で出荷された。

    ここでは訳した段落について「登録簿のいずれかの語に当たった段落の割合」を出す。
    これは完全な物差しではない —— 語の総数ではなく段落を数えているし、
    「登録すべき語をすべて登録したか」には答えない。**答えられないことは書かない。**
    """
    if not ja:
        return {"paragraphs": [0, 0], "terms": len(glossary), "untouched": []}
    by_key = {(s.chapter, s.index): s for s in src}
    hit = 0
    for r in ja:
        s = by_key.get((r.chapter, r.index))
        if s and any(term_pattern(t, e).search(s.text)
                     for t, e in glossary.items()):
            hit += 1
    # 訳した範囲に一度も現れない登録語(まだ先の章にしか出ない語)
    done = {(r.chapter, r.index) for r in ja}
    untouched = [t for t, e in glossary.items()
                 if not any(term_pattern(t, e).search(s.text)
                            for s in src if (s.chapter, s.index) in done)]
    return {"paragraphs": [hit, len(ja)], "terms": len(glossary),
            "untouched": sorted(untouched)}


def fill_rate(src: list[Source], ja: list[Rendered]) -> dict[str, object]:
    """**分子と分母で出す。** 「何割」だけを書かない。"""
    per_chapter: dict[int, list[int]] = {}
    for s in src:
        per_chapter.setdefault(s.chapter, [0, 0])[1] += 1
    for r in ja:
        if r.chapter in per_chapter:
            per_chapter[r.chapter][0] += 1
    return {
        "paragraphs": [len(ja), len(src)],
        "by_chapter": {str(k): v for k, v in sorted(per_chapter.items())},
        "chars": sum(len(r.text) for r in ja),
    }


def run() -> dict[str, object]:
    src = load_source()
    ja = load_translation()
    glossary = load_glossary()
    return {
        "fill": fill_rate(src, ja),
        "paragraph_errors": check_paragraphs(src, ja),
        "name_errors": check_names(src, ja, glossary),
        "number_errors": check_numbers(src, ja),
        "glossary_errors": check_glossary(src, ja, glossary),
        "length_outliers": check_lengths(src, ja),
        "registry_errors": check_registry_facts(src, glossary),
        "coverage": registry_coverage(src, ja, glossary),
        "terms": len(glossary),
    }


def main(argv=None) -> int:
    r = run()
    fill = r["fill"]
    n, d = fill["paragraphs"]
    print(f"自前和訳 {n}/{d} 段落({fill['chars']:,} 字)・登録簿 {r['terms']} 語")
    print("  章別(訳/原):", ", ".join(f"{k} 章 {v[0]}/{v[1]}"
                                       for k, v in fill["by_chapter"].items()))
    cov = r["coverage"]
    a, b = cov["paragraphs"]
    print(f"  登録簿の被覆: 訳した段落のうち {a}/{b} に登録語が現れる"
          f"(まだ出ていない登録語 {len(cov['untouched'])} 語)")
    bad = 0
    for key, label in (("paragraph_errors", "段落"), ("name_errors", "固有名"),
                       ("number_errors", "数"), ("glossary_errors", "訳語"),
                       ("length_outliers", "字数比の外れ"),
                       ("registry_errors", "登録簿の数字")):
        errs = r[key]
        bad += len(errs)
        print(f"  {label}: {'OK' if not errs else f'{len(errs)} 件'}")
        for e in errs[:20]:
            print(f"    - {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
