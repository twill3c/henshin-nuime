"""外挿検証(G-09)—— 『審判 / Der Prozess』で同じ手続きを回す。

**この一冊で出た結論が、この一冊限りのものかを見る。** 素材は同じ三者
(カフカ / David Wyllie 訳 / 原田義人訳)による別の長編である。

**権利は作品ごとに違う(HC-255)。** 実測した結果:

    独 PG #69327        COPYRIGHTED の表記なし。1925 年 Verlag die Schmiede。**公有**
    日 青空文庫 49863   原田義人訳。1960-08-01 没・旧法で 2010-12-31 満了。**公有**
    英 PG #7849         冒頭に COPYRIGHTED と明記。著作権者(David Wyllie)の
                        許諾で PG に収録されている作品。**公有ではない**

『変身』では英訳(PG #5200)が公有だったので、同じ訳者・同じ配布元なら同じだろうと
想定していた。**それが崩れた。** したがって:

- **英訳の本文は一字も配らない。** 公開面に載せるのは集計値だけである。
- 局所での計測は著作権法 30 条の 4(情報解析)の範囲で行う。
  求めているのは表現の享受ではなく、段落構造という統計量である。

『変身』の器をそのまま使い回さない。あちらは 3 章に決め打ちしてあり、
こちらは 10 章ある。**章の切り方は作品ごとに実測して決める**(§3 と同じ規律)。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import align, ingest, sentences
else:  # スクリプトとして直接起動されたとき
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import align, ingest, sentences

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "extrapolation"

WORK = "der_prozess"
EXPECTED_CHAPTERS = 10

# 独語の章見出し。**この本の綴りで書く** —— SIEBENTES(SIEBTES ではない)。
_DE_ORDINALS = ["ERSTES", "ZWEITES", "DRITTES", "VIERTES", "FÜNFTES",
                "SECHSTES", "SIEBENTES", "ACHTES", "NEUNTES", "ZEHNTES"]
_DE_HEAD = re.compile(r"(?m)^(" + "|".join(_DE_ORDINALS) + r")\s+KAPITEL\s*$")

_JA_ORDINALS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
_JA_HEAD = re.compile(r"第(" + "|".join(_JA_ORDINALS) + r")章")

# 英訳の章見出し。**本文は配らないが、構造を数えるためにここで切る。**
_EN_HEAD = re.compile(
    r"(?m)^Chapter (One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*$")


class ExtrapolationError(RuntimeError):
    """外挿の前提が崩れたときに投げる。黙って別の本を測らない。"""


@dataclass(frozen=True)
class Para:
    chapter: int
    index: int
    text: str


def _paras_from_chapters(chapters: list[list[str]], *,
                         drop_title: bool = False) -> list[Para]:
    """章ごとの段落列を通しの段落にする。

    `drop_title` は各章の先頭段落を落とす。**『審判』の各章には章題がある** ——
    独語なら `VERHAFTUNG · GESPRÄCH MIT FRAU GRUBACH · DANN FRÄULEIN BÜRSTNER`、
    英訳なら `Arrest--Conversation with Mrs. Grubach--Then Miss Bürstner`。
    『変身』の章には題が無かったので、この扱いを決めていなかった(実際に一度
    章題を本文として数え、独英の段落数が食い違った)。日本語版は章題が見出しの
    中にあるので、こちらは落とすものが無い。
    """
    out: list[Para] = []
    for c, paras in enumerate(chapters, start=1):
        body = paras[1:] if drop_title and paras else paras
        for i, t in enumerate(body, start=1):
            out.append(Para(c, i, t))
    return out


def load_de() -> list[Para]:
    raw = (RAW / f"de_pg69327.txt").read_text(encoding="utf-8")
    body = ingest.strip_pg_wrapper(raw)
    # **後書きは本文ではない。** 独語版には Max Brod の NACHWORT が続いており、
    # 英訳版には無い。切らずに数えると独語だけ 24 段落多くなる(実際に踏んだ)。
    cut = body.find("\nNACHWORT")
    if cut < 0:
        raise ExtrapolationError("NACHWORT が見つからない — 版が差し替わった可能性")
    body = body[:cut]
    marks = list(_DE_HEAD.finditer(body))
    if len(marks) != EXPECTED_CHAPTERS:
        raise ExtrapolationError(f"独語の章見出しが {len(marks)} 個")
    if [m.group(1) for m in marks] != _DE_ORDINALS:
        raise ExtrapolationError("独語の章見出しの並びが期待と違う")
    chapters = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        seg = body[m.end():end]
        paras = [ingest.normalize(b) for b in re.split(r"\n[ \t]*\n", seg)]
        chapters.append([p for p in paras if p])
    return _paras_from_chapters(chapters, drop_title=True)


def load_en() -> list[Para]:
    """**本文を配らない。** 段落構造を数えるためだけに読む(HC-255)。"""
    raw = (RAW / "en_pg7849.txt").read_text(encoding="utf-8")
    if "COPYRIGHTED" not in raw:
        raise ExtrapolationError(
            "英訳の COPYRIGHTED 表記が消えている —— 権利の前提が変わった。"
            "配らない決まりを見直す前に、条項を読み直すこと")
    body = ingest.strip_pg_wrapper(raw)
    marks = list(_EN_HEAD.finditer(body))
    if len(marks) != EXPECTED_CHAPTERS:
        raise ExtrapolationError(f"英訳の章見出しが {len(marks)} 個")
    chapters = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        seg = body[m.end():end]
        paras = [ingest.normalize(b) for b in re.split(r"\n[ \t]*\n", seg)]
        chapters.append([p for p in paras if p])
    return _paras_from_chapters(chapters, drop_title=True)


def load_ja() -> list[Para]:
    raw = (RAW / "ja_aozora49863.html").read_bytes().decode("shift_jis")
    if '<div class="main_text">' not in raw:
        raise ExtrapolationError("main_text が見つからない")
    main = raw.split('<div class="main_text">', 1)[1]
    main = main.split('<div class="bibliographical_information">', 1)[0]

    # 見出しは中見出しの中に「第N章」として入っている。**外字画像ではない。**
    marks = [m for m in re.finditer(r'<h[45][^>]*>(.*?)</h[45]>', main, re.S)
             if _JA_HEAD.search(m.group(1))]
    if len(marks) != EXPECTED_CHAPTERS:
        raise ExtrapolationError(f"日本語の章見出しが {len(marks)} 個")
    order = [_JA_HEAD.search(m.group(1)).group(1) for m in marks]
    if order != _JA_ORDINALS:
        raise ExtrapolationError(f"日本語の章見出しの並びが {order}")

    chapters = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(main)
        seg = main[m.end():end]
        paras = [ingest.strip_aozora_markup(b) for b in re.split(r"<br\s*/?>", seg)]
        chapters.append([p for p in paras if p])
    return _paras_from_chapters(chapters)


def chapter_counts(paras: list[Para]) -> list[int]:
    counts = [0] * EXPECTED_CHAPTERS
    for p in paras:
        counts[p.chapter - 1] += 1
    return counts


def to_sentences(paras: list[Para], lang: str) -> list[sentences.Sentence]:
    """段落を文に切る。**規則は『変身』で決めたものをそのまま使う。**

    ここで規則を作品に合わせて調整すると、外挿が外挿でなくなる ——
    「別の本でも効くか」を見たいのだから、当てはめ直してはいけない。
    """
    out: list[sentences.Sentence] = []
    for p in paras:
        for span in sentences.spans(p.text, lang):
            out.append(sentences.Sentence(
                sid=f"{WORK}_{lang}-{p.chapter}-{p.index}-{len(out)}",
                edition_id=f"{WORK}_{lang}",
                chapter=p.chapter,
                paragraph=p.index,
                index=len(out),
                text=p.text[span[0]:span[1]],
            ))
    return out


def structure() -> dict:
    """三版の構造を数える。**英訳は本文を持ち出さず、数だけを出す。**"""
    de, en, ja = load_de(), load_en(), load_ja()
    out = {}
    for key, paras, lang in (("de", de, "de"), ("en", en, "en"), ("ja", ja, "ja")):
        sents = to_sentences(paras, lang)
        out[key] = {
            "paragraphs": len(paras),
            "chapter_paragraphs": chapter_counts(paras),
            "chars": sum(len(p.text) for p in paras),
            "sentences": len(sents),
        }
    return out


def embed_all() -> dict[str, object]:
    """外挿セットの埋め込みを焼く。**独語と日本語だけ。**

    英訳は本文を配らないので、公開面が読む埋め込みも作らない ——
    埋め込みから原文は復元できないが、**配らないと決めたものの派生物を
    配る道を作らない**。段落構造の比較は文字数だけで足りる。
    """
    import numpy as np

    from . import embed as embed_mod

    OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, object] = {}
    for key, load, lang in (("de", load_de, "de"), ("ja", load_ja, "ja")):
        path = OUT / f"emb_{key}.npy"
        sents = to_sentences(load(), lang)
        if path.exists():
            vecs = np.load(path)
            if len(vecs) != len(sents):
                raise ExtrapolationError(
                    f"{key}: 焼いた埋め込み {len(vecs)} 本に対し文 {len(sents)} 件")
        else:
            print(f"{key}: {len(sents)} 文を埋め込む…", flush=True)
            vecs = embed_mod.embed_texts([s.text for s in sents])
            np.save(path, vecs)
        out[key] = vecs
    return out


def align_all() -> dict:
    """三手法で独→日を縫い、『変身』と同じ物差しで測る。

    **規則も閾値も当てはめ直さない。** 分割規則は『変身』で決めたもの、
    Gale & Church の分散も論文値のまま。当てはめ直せば外挿にならない。
    """
    import numpy as np

    from . import evaluate

    de = to_sentences(load_de(), "de")
    ja = to_sentences(load_ja(), "ja")
    vecs = embed_all()

    src_t = [s.text for s in de]
    dst_t = [s.text for s in ja]
    c = align.stretch_from_totals(src_t, dst_t)
    chance = align.chance_similarity(vecs["de"], vecs["ja"])
    methods = {
        "gale_church": align.align(src_t, dst_t, c=c),
        "diagonal": align.align_diagonal(len(src_t), len(dst_t)),
        "embedding": align.align_embeddings(vecs["de"], vecs["ja"], baseline=chance),
    }
    rows: dict[str, object] = {"c": c, "chance": float(chance),
                               "n_src": len(de), "n_dst": len(ja)}
    for name, beads in methods.items():
        ref = evaluate.refinement_violations(beads, de, ja)
        cov_src, cov_dst = align.coverage(beads, len(de), len(ja))
        rows[name] = {
            "links": len(align.links(beads)),
            "refinement_violations": ref.violations,
            "refinement_paragraphs": ref.paragraphs,
            "coverage_src": cov_src,
            "coverage_dst": cov_dst,
            "shapes": {f"{k[0]}-{k[1]}": v
                       for k, v in sorted(align.shape_counts(beads).items())},
        }

    # **被覆が手法ごとに違うので、そのまま並べない**(§3.4 と同じ規律)。
    # 埋め込みは偶然の水準を超えない対応を組まないぶん日本語側を飛ばすので、
    # 飛ばすほど破れが減って見える。全手法が覆った段落だけで測り直す。
    from . import stats

    names = ["embedding", "gale_church", "diagonal"]
    common = set.intersection(*(
        evaluate.covered_dst_paragraphs(methods[n], ja) for n in names))
    rows["common_dst_paragraphs"] = len(common)
    for n in names:
        ref = evaluate.refinement_violations(methods[n], de, ja, dst_subset=common)
        rows[n]["refinement_violations_common"] = ref.violations
        rows[n]["refinement_paragraphs_common"] = ref.paragraphs

    # 差が偶然で説明できるかを、『変身』と同じ検定に掛ける。
    # ブロックの大きさは構造から決まらないので**一つに決めない**(G-20)。
    units = sorted(common)
    base = evaluate.refinement_outcomes(methods["embedding"], de, ja, common)
    tests: dict[str, object] = {}
    for other in ("gale_church", "diagonal"):
        comp = evaluate.refinement_outcomes(methods[other], de, ja, common)
        for size in evaluate.REFINEMENT_BLOCK_SIZES:
            blocks = [k // size for k in range(len(units))]
            r = stats.paired_permutation([base[p] for p in units],
                                         [comp[p] for p in units], blocks)
            tests[f"refinement_vs_{other}_block{size}"] = {
                "difference": r.observed, "p": r.p_display}
    rows["tests"] = tests
    return rows


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st = structure()
    print(f"『審判』の構造(外挿検証・{WORK})")
    for key, v in st.items():
        tag = "(本文は配らない)" if key == "en" else ""
        print(f"  {key}: 段落 {v['paragraphs']:4} / 文 {v['sentences']:5} / "
              f"{v['chars']:,} 字 {tag}")
        print(f"      章別 {v['chapter_paragraphs']}")
    same = st["de"]["paragraphs"] == st["en"]["paragraphs"]
    print()
    print(f"独英の段落数一致: {'はい' if same else 'いいえ'}"
          f"({st['de']['paragraphs']} 対 {st['en']['paragraphs']})")
    print(f"章別も一致: "
          f"{'はい' if st['de']['chapter_paragraphs'] == st['en']['chapter_paragraphs'] else 'いいえ'}")
    (OUT / "structure.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
