"""起動経路の検査。TEST_SPEC の T-015 に対応する。

テストは pipeline を package として import するので、**CLI が死んでいても
すべて緑になる**。実際に L0 でそれが起きた(相対 import で ImportError、
それでも pytest は 18 件全緑)。だから起動そのものを一度走らせる。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )


# 二つの起動経路を両方見る。片方だけを守ると、動くほうを覚えた次の人が
# 動かないほうを踏む(L0 で実際に起きた)。
ENTRYPOINTS = [
    ("-m", "pipeline.ingest"),
    ("-m", "pipeline.rights"),
    ("-m", "pipeline.sentences"),
    ("-m", "pipeline.evaluate"),
    # 全文の埋め込みは 16 分かかるので、起動経路の確認は軽い方を通す。
    # 検査に 16 分かかる工程を入れると、その検査は回されなくなる。
    ("-m", "pipeline.embed", "--stats"),
    ("pipeline/ingest.py",),
    ("pipeline/rights.py",),
    ("pipeline/sentences.py",),
    ("pipeline/evaluate.py",),
    ("pipeline/embed.py", "--stats"),
]

# モデルを要する起動経路は、手元にモデルが無ければ飛ばす(約 940 MB で git に入れていない)
NEEDS_MODEL = {"pipeline.embed", "pipeline/embed.py"}
MODEL_FILES = ("tokenizer.json",)

# `pipeline/align.py` は本文を読む口を持たないので起動経路が無い(G-03)。
# 「入口が無いこと」自体を固定しておかないと、後から誰かが main を足したときに
# 段落を知る型への経路が静かに開く。
NO_ENTRYPOINT = ["pipeline/align.py"]


@pytest.mark.validation
@pytest.mark.parametrize("args", ENTRYPOINTS, ids=lambda a: " ".join(a))
def test_t015_cli_entrypoints_run(args):
    """T-015 / T-036 — 起動経路が終了コード 0 で完走する。"""
    if set(args) & NEEDS_MODEL:
        missing = [f for f in MODEL_FILES
                   if not (ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
                           / f).exists()]
        if missing:
            pytest.skip(f"モデル {missing} が手元に無い(README の入手手順を見ること)")
    proc = _run(*args)
    assert proc.returncode == 0, (
        f"{' '.join(args)} が終了コード {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert proc.stdout.strip(), f"{' '.join(args)} が何も出力していない"


@pytest.mark.validation
@pytest.mark.parametrize("path", NO_ENTRYPOINT)
def test_t030_modules_without_entrypoint_stay_that_way(path):
    """T-030 — 入口を持たないと決めたモジュールに入口が生えていないこと。"""
    source = (ROOT / path).read_text(encoding="utf-8")
    assert '__main__' not in source, f"{path} に起動経路が生えている(G-03)"


@pytest.mark.validation
def test_t015_positive_control():
    """陽性対照 — 起動に失敗するものは実際に落ちること。

    これが無いと、`_run` が常に成功を返す実装でも上のケースは緑になる。
    """
    assert _run("-m", "pipeline.this_module_does_not_exist").returncode != 0
    assert _run("pipeline/this_file_does_not_exist.py").returncode != 0
