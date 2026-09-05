"""四版の生ファイルを、章 → 段落の共通構造に落とす取り込み器。

方針(AGENTS.md「外部データの採録」):

- 分割規則はこのモジュールにだけ書く。探索用に別途書き写さない(HC-069)。
- 仮定が崩れたら黙って違う結果を出さず、その場で例外にする(HC-075)。
  章数・境界標識・本文の在存はすべて assert ではなく IngestError で落とす。
- 段落は「アラインメントの評価に使う非循環オラクル」の素になるため、
  ここでの分割規則は言語・版に依存しない同じ述語で書く。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "editions"

EXPECTED_CHAPTERS = 3


class IngestError(RuntimeError):
    """外部ファイルの構造が仮定と食い違ったときに投げる。"""


@dataclass(frozen=True)
class Paragraph:
    pid: str
    chapter: int
    index: int
    text: str


@dataclass
class Edition:
    edition_id: str
    lang: str
    title: str
    author: str
    translator: str | None
    source_url: str
    paragraphs: list[Paragraph]

    def to_json(self) -> dict:
        d = asdict(self)
        d["paragraph_count"] = len(self.paragraphs)
        d["chapter_paragraph_counts"] = chapter_counts(self.paragraphs)
        return d


def chapter_counts(paragraphs: list[Paragraph]) -> list[int]:
    counts: list[int] = []
    for p in paragraphs:
        while len(counts) < p.chapter:
            counts.append(0)
        counts[p.chapter - 1] += 1
    return counts


def normalize(text: str) -> str:
    """行内改行を潰し、Unicode 正規化する。字面の書き換えはここでは行わない。

    引用符・ダッシュの字種は版ごとの事実なので保存する。
    正規化で潰すと、独 »« と英 "" の対応という構造オラクルが消える。
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- Project Gutenberg -------------------------------------------------------

_PG_START = re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*$", re.M)
_PG_END = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*$", re.M)


def strip_pg_wrapper(raw: str) -> str:
    """PG のヘッダ・フッタを落とし、本文だけを返す。

    PG の許諾表示と商標はこの時点で本文から切り離す。出典は権利台帳が持つ。
    """
    m_start = _PG_START.search(raw)
    m_end = _PG_END.search(raw)
    if m_start is None or m_end is None:
        raise IngestError("PG の START/END 標識が見つからない — 版が差し替わった可能性")
    if m_end.start() <= m_start.end():
        raise IngestError("PG の END 標識が START より前にある")
    return raw[m_start.end() : m_end.start()]


def split_pg_chapters(body: str) -> list[list[str]]:
    """PG 本文を章に切り、各章を空行区切りの段落に分ける。

    章標識は行全体がローマ数字(直後の句点は任意)である行に限る。
    本文中に同じ形の行が現れると誤爆するので、見つかった数を必ず検算する。
    """
    marks = list(re.finditer(r"(?m)^[ \t]*(I{1,3})\.?[ \t]*$", body))
    if len(marks) != EXPECTED_CHAPTERS:
        raise IngestError(
            f"章標識が {len(marks)} 個 — {EXPECTED_CHAPTERS} 個を期待"
        )
    numerals = [m.group(1) for m in marks]
    if numerals != ["I", "II", "III"]:
        raise IngestError(f"章標識の並びが {numerals} — I/II/III を期待")

    chapters: list[list[str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        seg = body[m.end() : end]
        paras = [normalize(b) for b in re.split(r"\n[ \t]*\n", seg)]
        chapters.append([p for p in paras if p])
    return chapters


# --- 青空文庫 ----------------------------------------------------------------

_AOZORA_RUBY = re.compile(
    r"<ruby><rb>(.*?)</rb><rp>.*?</rp><rt>(.*?)</rt><rp>.*?</rp></ruby>", re.S
)
_AOZORA_HEADING = re.compile(
    r'<h4 class="naka-midashi">.*?<img[^>]*alt="※\(ローマ数字([123])[^"]*"[^>]*/?>.*?</h4>',
    re.S,
)


def strip_aozora_markup(fragment: str) -> str:
    """ルビの親文字だけを残し、残りのタグを落とす。

    ルビ(振仮名)は本文の一部ではなく注記なので、本文からは外す。
    落とした振仮名は extract_aozora_ruby が別に拾う。
    """
    fragment = _AOZORA_RUBY.sub(lambda m: m.group(1), fragment)
    if "<img" in fragment:
        raise IngestError("本文中に外字画像が残った — alt の取り扱いを決めていない")
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return normalize(fragment)


def extract_aozora_ruby(raw: str) -> list[tuple[str, str]]:
    """(親文字, 振仮名) の一覧。取りこぼしの検算に使う。"""
    return [(m.group(1), m.group(2)) for m in _AOZORA_RUBY.finditer(raw)]


def split_aozora_chapters(raw: str) -> list[list[str]]:
    """青空文庫 XHTML の main_text を章 → 段落に切る。

    青空の段落は <p> ではなく <br /> 区切りの行である。
    見出しは jisage_5 の h4 に外字画像で入っているので、alt から番号を読む。
    """
    if '<div class="main_text">' not in raw:
        raise IngestError("main_text が見つからない — 青空の書式が変わった可能性")
    main = raw.split('<div class="main_text">', 1)[1]
    main = main.split('<div class="bibliographical_information">', 1)[0]

    marks = list(_AOZORA_HEADING.finditer(main))
    if len(marks) != EXPECTED_CHAPTERS:
        raise IngestError(f"章見出しが {len(marks)} 個 — {EXPECTED_CHAPTERS} 個を期待")
    numerals = [m.group(1) for m in marks]
    if numerals != ["1", "2", "3"]:
        raise IngestError(f"章見出しの並びが {numerals} — 1/2/3 を期待")

    chapters: list[list[str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(main)
        seg = main[m.end() : end]
        paras = [strip_aozora_markup(b) for b in re.split(r"<br\s*/?>", seg)]
        chapters.append([p for p in paras if p])
    return chapters


# --- 組み立て ----------------------------------------------------------------


def build(edition_id: str, lang: str, title: str, author: str,
          translator: str | None, source_url: str,
          chapters: list[list[str]]) -> Edition:
    paragraphs: list[Paragraph] = []
    for ci, paras in enumerate(chapters, start=1):
        if not paras:
            raise IngestError(f"{edition_id} 第{ci}章の段落が 0 件")
        for pi, text in enumerate(paras, start=1):
            paragraphs.append(
                Paragraph(pid=f"{edition_id}-{ci}-{pi:03d}", chapter=ci,
                          index=pi, text=text)
            )
    return Edition(edition_id, lang, title, author, translator, source_url,
                   paragraphs)


def load_de(raw_dir: Path = RAW_DIR) -> Edition:
    raw = (raw_dir / "de_pg22367.txt").read_text(encoding="utf-8")
    chapters = split_pg_chapters(strip_pg_wrapper(raw))
    return build("de_pg22367", "de", "Die Verwandlung", "Franz Kafka", None,
                 "https://www.gutenberg.org/ebooks/22367", chapters)


def load_en(raw_dir: Path = RAW_DIR) -> Edition:
    raw = (raw_dir / "en_pg5200.txt").read_text(encoding="utf-8")
    chapters = split_pg_chapters(strip_pg_wrapper(raw))
    return build("en_pg5200", "en", "Metamorphosis", "Franz Kafka",
                 "David Wyllie", "https://www.gutenberg.org/ebooks/5200",
                 chapters)


def load_ja(raw_dir: Path = RAW_DIR) -> Edition:
    raw = (raw_dir / "ja_aozora49866.html").read_bytes().decode("shift_jis")
    chapters = split_aozora_chapters(raw)
    return build("ja_aozora49866", "ja", "変身", "フランツ・カフカ", "原田義人",
                 "https://www.aozora.gr.jp/cards/001235/card49866.html",
                 chapters)


LOADERS = {"de_pg22367": load_de, "en_pg5200": load_en, "ja_aozora49866": load_ja}


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, Edition]:
    return {k: fn(raw_dir) for k, fn in LOADERS.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for eid, ed in load_all().items():
        path = OUT_DIR / f"{eid}.json"
        path.write_text(
            json.dumps(ed.to_json(), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"{eid}: 段落 {len(ed.paragraphs)} 件 "
              f"章別 {chapter_counts(ed.paragraphs)} → {path.name}")


if __name__ == "__main__":
    main()
