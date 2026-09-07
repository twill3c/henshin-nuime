"""検査全体で共有する重い成果物。

`evaluate.run()` は三組の DP と置換検定で 2 分ほどかかる。**モジュールをまたいで
一度だけ計算する**ため、session スコープのフィクスチャはここに置く
(module ごとに定義すると共有されない)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import evaluate, sentences  # noqa: E402


@pytest.fixture(scope="session")
def sents():
    return sentences.load_all()


@pytest.fixture(scope="session")
def results():
    return evaluate.run()
