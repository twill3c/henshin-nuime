"""権利台帳の読み込みと検査(G-01)。

台帳は公開面の出典表示の正本でもある。ここで欠落を落とさないと、
「出典を書いたつもり」のまま公開まで行く。

青空文庫の取り扱い規準(2026-09-05 取得)は、著作権の切れた作品について
自由な複製・再配布を認めるかわりに、**作品名・著者名・翻訳者名・底本・入力者・
校正者・日付の情報が削除されないこと**を求めている。したがって青空由来の版には
一般の版より多くの必須欄を課す。

起動は `python -m pipeline.rights` でも `python pipeline/rights.py` でもよい。
**どちらか片方だけが動く状態を残さない** —— `ingest.py` は同一パッケージへの
import を持たないためスクリプト起動でも動いてしまい、`rights.py` だけが
落ちる状態は次に触る者への罠になる。T-015 が両方の経路を実際に走らせる。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__:
    from . import ingest
else:  # スクリプトとして直接起動されたとき。sys.path[0] は pipeline/ になる
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import ingest

LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "rights.json"

AOZORA = "青空文庫"

# すべての版に要る欄
REQUIRED = (
    "edition_id", "lang", "title", "author", "source_name", "source_url",
    "retrieved", "basis", "terms_url",
)
REQUIRED_LISTS = ("attribution", "obligations")

# 青空由来の版にだけ要る欄(取り扱い規準の記載事項)
REQUIRED_AOZORA = (
    "base_text", "publisher", "first_published", "input_by", "proofread_by",
    "created", "revised",
)


def load_ledger(path: Path = LEDGER_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get(ledger: dict, edition_id: str) -> dict:
    for e in ledger["editions"]:
        if e["edition_id"] == edition_id:
            return e
    raise KeyError(edition_id)


def check_ledger(ledger: dict) -> list[str]:
    """台帳の不備を並べて返す。不備が無ければ空リスト。"""
    problems: list[str] = []

    editions = ledger.get("editions")
    if not editions:
        return ["editions が空"]

    ids = [e.get("edition_id") for e in editions]
    if len(ids) != len(set(ids)):
        problems.append(f"edition_id が重複している: {ids}")
    missing = set(ingest.LOADERS) - set(ids)
    if missing:
        problems.append(f"取り込み対象なのに台帳に無い版: {sorted(missing)}")
    extra = set(ids) - set(ingest.LOADERS)
    if extra:
        problems.append(f"台帳にあるが取り込み対象でない版: {sorted(extra)}")

    for e in editions:
        eid = e.get("edition_id", "(id 無し)")
        if "translator" not in e:
            problems.append(f"{eid}: translator の欄が無い(訳者なしなら null を置く)")
        for field in REQUIRED:
            if not e.get(field):
                problems.append(f"{eid}: {field} が空")
        for field in REQUIRED_LISTS:
            value = e.get(field)
            if not isinstance(value, list) or not value or not all(value):
                problems.append(f"{eid}: {field} が空")
        if e.get("source_name") == AOZORA:
            if not e.get("translator"):
                problems.append(f"{eid}: 翻訳者名が空(青空の記載事項)")
            for field in REQUIRED_AOZORA:
                if not e.get(field):
                    problems.append(f"{eid}: {field} が空(青空の記載事項)")

    return problems


def main() -> None:
    ledger = load_ledger()
    problems = check_ledger(ledger)
    for p in problems:
        print("NG:", p)
    print(f"版 {len(ledger['editions'])} 件 / 不備 {len(problems)} 件")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
