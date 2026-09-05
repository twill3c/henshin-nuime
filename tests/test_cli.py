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
    ("pipeline/ingest.py",),
    ("pipeline/rights.py",),
    ("pipeline/sentences.py",),
]


@pytest.mark.validation
@pytest.mark.parametrize("args", ENTRYPOINTS, ids=lambda a: " ".join(a))
def test_t015_cli_entrypoints_run(args):
    """T-015 — 起動経路が終了コード 0 で完走する。"""
    proc = _run(*args)
    assert proc.returncode == 0, (
        f"{' '.join(args)} が終了コード {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert proc.stdout.strip(), f"{' '.join(args)} が何も出力していない"


@pytest.mark.validation
def test_t015_positive_control():
    """陽性対照 — 起動に失敗するものは実際に落ちること。

    これが無いと、`_run` が常に成功を返す実装でも上のケースは緑になる。
    """
    assert _run("-m", "pipeline.this_module_does_not_exist").returncode != 0
    assert _run("pipeline/this_file_does_not_exist.py").returncode != 0
