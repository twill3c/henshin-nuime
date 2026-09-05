"""段落を文に切る。三言語それぞれの規則は、**その言語の本文だけを見て**決めた。

**非循環の規律(G-15)。** 文の数が言語間で揃うことは、この企画が測ろうとしている当のもの
である。だから分割規則を「揃うように」調整してはならない。ここに書いた規則はすべて、
2026-09-05 に各版を単独で走査して出た現象から導いた。他言語の分割結果を見て足した規則は
一つも無く、今後も足さない。揃わなければ揃わないまま出す。

**取りこぼしを許さない(G-14)。** 分割は境界位置の列を返すだけで、文はその区間の切り出し
である。区間の外に落ちるのは前後の空白だけであり、`spans()` の結果から元の段落を復元できる。
復元できなければ例外にする —— 「未分類 0 件」を数える前に、そもそも取りこぼしが起きない
形にしてある。

各言語の実測(2026-09-05、母集団は各版の本文全段落):

独 `de_pg22367`
  終止符 + 空白 の候補 629 件。うち直後が小文字 27 件 —— **全件が閉じ « を伴う**。
  これは »…?« dachte er. の形で、引用が後続節の目的語になっており文は終わっていない。
  直後がハイフン 3 件 —— `? --` で、ダッシュの挿入句の中にある疑問符。
  既知の独語略語(z. B. / usw. / Nr. / Dr. 等)は **0 件**。序数の「数字.」も 0 件。

英 `en_pg5200`
  候補 716 件。うち直後が小文字 11 件 —— 9 件は閉じ ” を伴う引用の後続節、
  2 件が `Mr.`。**略語 `Mr.` 21 件・`Mrs.` 10 件があり、その 31 件すべてで直後が大文字**。
  素朴な分割器はここで黙って切る。三点リーダ `...` は 1 件(段落末)。

日 `ja_aozora49866`
  `」` 129 件。直前に `。` が来る例は **0 件**(句点を閉じ括弧の前に置かない組版)。
  `」` の直後は と 84 件・段落末 37 件・その他 8 件。その他 8 件のうち継続は
  `など`(引用の助詞)1 件のみで、残る 7 件は新しい文の開始。
  `。` の直後が と である例が 100 件あるが、いずれも「というのは」「ところで」など
  と で始まる語であって、`。` は例外なく文末である。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# 起動は `python -m pipeline.sentences` でも `python pipeline/sentences.py` でもよい。
# **どちらか片方だけが動く状態を残さない**(HC-174)。T-015 が両方を実際に走らせる。
if __package__:
    from . import ingest
else:  # スクリプトとして直接起動されたとき。sys.path[0] は pipeline/ になる
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import ingest

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sentences"


class SplitError(RuntimeError):
    """分割が取りこぼしを起こした、または想定外の入力に当たったときに投げる。"""


@dataclass(frozen=True)
class Sentence:
    sid: str
    edition_id: str
    chapter: int
    paragraph: int
    index: int
    text: str


# --- 独語 --------------------------------------------------------------------

_DE = re.compile(r"[.!?]«?")


def _spans_de(text: str, rules: bool = True) -> list[int]:
    """`rules=False` は言語ごとの例外規則を外した対照。

    対照を**テスト側で書き直さない**ためにここに置く。書き写すと、
    対照だけが古い規則のまま緑になる(HC-069)。
    """
    ends: list[int] = []
    for m in _DE.finditer(text):
        end = m.end()
        rest = text[end:]
        if not rest:
            continue  # 段落末は最後にまとめて閉じる
        if not rest[:1].isspace():
            continue  # 語中の記号(略語・数字)は境界にしない
        nxt = rest.lstrip()[:1]
        if not nxt:
            continue
        if rules:
            if nxt.islower():
                continue  # »…?« dachte er. — 引用が後続節の目的語
            if nxt == "-":
                continue  # `? --` — ダッシュの挿入句の中
        ends.append(end)
    return ends


# --- 英語 --------------------------------------------------------------------

_EN = re.compile(r"[.!?]+[”’]?")
# 略語は英語の本文だけを走査して見つけた(2026-09-05)。
# **他言語を見て足したものは無い。**
EN_ABBREVIATIONS = ("Mr.", "Mrs.")


def _spans_en(text: str, rules: bool = True) -> list[int]:
    ends: list[int] = []
    for m in _EN.finditer(text):
        end = m.end()
        rest = text[end:]
        if not rest or not rest[:1].isspace():
            continue
        head = text[:end]
        if rules and any(head.endswith(a) for a in EN_ABBREVIATIONS):
            continue
        nxt = rest.lstrip()[:1]
        if not nxt or nxt.islower():
            continue
        ends.append(end)
    return ends


# --- 日本語 ------------------------------------------------------------------

# 引用を受ける助詞。`「…」と、彼は思った。` の と がこれにあたる。
JA_QUOTATIVE = ("と", "など")


def _spans_ja(text: str, rules: bool = True) -> list[int]:
    ends: list[int] = []
    for i, ch in enumerate(text):
        end = i + 1
        rest = text[end:]
        if ch == "。":
            pass
        elif ch in "！？":
            if rest[:1] == "」":
                continue  # 閉じ括弧まで含めてから切る
        elif ch == "」":
            if rules and any(rest.startswith(q) for q in JA_QUOTATIVE):
                continue  # 引用を受ける助詞が続く — 文は終わっていない
        else:
            continue
        if not rest.strip():
            continue  # 段落末は最後にまとめて閉じる
        ends.append(end)
    return ends


_SPLITTERS = {"de": _spans_de, "en": _spans_en, "ja": _spans_ja}


def spans(text: str, lang: str, rules: bool = True) -> list[tuple[int, int]]:
    """文の区間を [(start, end), ...] で返す。区間の外は空白だけになる。

    `rules=False` は言語ごとの例外規則を外した対照で、**テスト専用ではない** ——
    SPEC §3.1 の「規則の寄与」表はこの経路で測っている。
    """
    if lang not in _SPLITTERS:
        raise SplitError(f"未知の言語: {lang}")
    ends = _SPLITTERS[lang](text, rules)
    out: list[tuple[int, int]] = []
    prev = 0
    for e in ends + [len(text)]:
        seg = text[prev:e]
        stripped = seg.strip()
        if stripped:
            start = prev + (len(seg) - len(seg.lstrip()))
            out.append((start, start + len(stripped)))
        prev = e
    if not out:
        raise SplitError("文が 1 つも取れなかった")
    check_lossless(text, out)
    return out


def check_lossless(text: str, sp: list[tuple[int, int]]) -> None:
    """区間の外に空白以外が落ちていないことを確かめる(G-14)。

    「未分類 0 件」を数えるのではなく、**取りこぼしが起きたら落ちる**ようにする。
    """
    cursor = 0
    for start, end in sp:
        gap = text[cursor:start]
        if gap.strip():
            raise SplitError(f"区間の外に本文が落ちた: {gap!r}")
        if start >= end:
            raise SplitError(f"空の区間: ({start}, {end})")
        cursor = end
    tail = text[cursor:]
    if tail.strip():
        raise SplitError(f"末尾を取りこぼした: {tail!r}")


def split(text: str, lang: str, rules: bool = True) -> list[str]:
    return [text[s:e] for s, e in spans(text, lang, rules)]


def count_for(edition: ingest.Edition, rules: bool = True) -> int:
    """その版の文の総数。`rules=False` で例外規則を外した対照が取れる。"""
    return sum(len(spans(p.text, edition.lang, rules)) for p in edition.paragraphs)


def sentences_for(edition: ingest.Edition) -> list[Sentence]:
    out: list[Sentence] = []
    for p in edition.paragraphs:
        for i, s in enumerate(split(p.text, edition.lang), start=1):
            out.append(
                Sentence(
                    sid=f"{p.pid}-{i:02d}",
                    edition_id=edition.edition_id,
                    chapter=p.chapter,
                    paragraph=p.index,
                    index=i,
                    text=s,
                )
            )
    return out


def load_all() -> dict[str, list[Sentence]]:
    return {eid: sentences_for(ed) for eid, ed in ingest.load_all().items()}


def chapter_counts(sents: list[Sentence]) -> list[int]:
    counts = [0, 0, 0]
    for s in sents:
        counts[s.chapter - 1] += 1
    return counts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for eid, sents in load_all().items():
        path = OUT_DIR / f"{eid}.json"
        path.write_text(
            json.dumps(
                {
                    "edition_id": eid,
                    "sentence_count": len(sents),
                    "chapter_sentence_counts": chapter_counts(sents),
                    "sentences": [asdict(s) for s in sents],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"{eid}: 文 {len(sents)} 件 章別 {chapter_counts(sents)} → {path.name}")


if __name__ == "__main__":
    main()
