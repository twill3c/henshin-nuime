/**
 * 図の座標まわりの検査(TEST_SPEC の T-061)。
 *
 * **図に添える文字は、図を描いたのと同じデータから導く**(HC-045)。
 * 位置も内容も決め打ちしないので、ここで確かめるのは
 * 「同じ入力から同じ座標が出る」ことと「端で壊れない」ことである。
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { linear, paragraphStarts, pathPoints, polyline, ticks } from "../src/core/figure";

describe("尺度", () => {
  it("両端が指定した範囲に一致する", () => {
    const s = linear(10, 0, 100);
    expect(s(0)).toBe(0);
    expect(s(9)).toBe(100);
    expect(s(4.5)).toBeCloseTo(50, 6);
  });

  it("要素が 1 個以下でも落ちず、中央を返す", () => {
    // 端の頁では起こりうる。**0 除算で NaN を返すと、図が黙って消える。**
    expect(linear(1, 10, 20)(0)).toBe(15);
    expect(linear(0, 10, 20)(0)).toBe(15);
    expect(Number.isNaN(linear(1, 10, 20)(0))).toBe(false);
  });

  it("上下を逆にした範囲でも単調", () => {
    const s = linear(5, 100, 0); // 縦軸は上が大きい
    expect(s(0)).toBe(100);
    expect(s(4)).toBe(0);
    expect(s(1)).toBeGreaterThan(s(2));
  });
});

describe("段落の切れ目", () => {
  it("範囲の中で 0 起点の位置を返す", () => {
    //          index: 0  1  2  3  4  5
    const paragraph = [7, 7, 8, 8, 8, 9];
    expect(paragraphStarts(paragraph, 0, 6)).toEqual([0, 2, 5]);
  });

  it("範囲の途中から始めても 0 起点になる", () => {
    const paragraph = [7, 7, 8, 8, 8, 9];
    expect(paragraphStarts(paragraph, 2, 6)).toEqual([0, 3]);
  });

  it("空の範囲では空", () => {
    expect(paragraphStarts([1, 2, 3], 1, 1)).toEqual([]);
  });
});

describe("目盛り", () => {
  it("端が必ず入る", () => {
    for (const n of [2, 7, 40, 275, 1227]) {
      const t = ticks(n, 8);
      expect(t[0]).toBe(0);
      expect(t[t.length - 1]).toBe(n - 1);
    }
  });

  it("本数が軸の長さに応じて増減する", () => {
    // 決め打ちだと短い軸で詰まり、長い軸で間延びする
    expect(ticks(1000, 8).length).toBeGreaterThan(3);
    expect(ticks(1000, 8).length).toBeLessThan(20);
  });

  it("要素が 1 個なら 1 本", () => {
    expect(ticks(1, 8)).toEqual([0]);
  });
});

describe("経路の点列", () => {
  it("同じ src に複数の dst があれば中央を取る", () => {
    // 端を取ると経路が実際より上か下に寄って見える
    expect(pathPoints([[0, 10], [0, 12], [0, 14]])).toEqual([[0, 12]]);
    expect(pathPoints([[0, 10], [0, 12]])).toEqual([[0, 10]]);
  });

  it("src の順に並ぶ", () => {
    const got = pathPoints([[5, 5], [1, 1], [3, 3]]);
    expect(got.map((p) => p[0])).toEqual([1, 3, 5]);
  });

  it("空なら空", () => {
    expect(pathPoints([])).toEqual([]);
  });
});

describe("折れ線", () => {
  it("点が無ければ空文字を返す", () => {
    // **空の d は描かれない。** "M" だけ書くと不正な path になる。
    expect(polyline([], (v) => v, (v) => v)).toBe("");
  });

  it("先頭が M で以降が L", () => {
    const d = polyline([[0, 0], [1, 2]], (v) => v * 10, (v) => v * 5);
    expect(d).toBe("M0.0 0.0 L10.0 10.0");
  });
});

const DATA = join(process.cwd(), "public", "data");
const maybe = existsSync(join(DATA, "attention.json")) ? describe : describe.skip;

maybe("注意のデータ", () => {
  const att = JSON.parse(readFileSync(join(DATA, "attention.json"), "utf-8"));

  it("値が [0,1] に収まり、帯の中だけに立つ", () => {
    const [n, m] = att.shape;
    expect(att.cells.length).toBeGreaterThan(0);
    // 全面の 5% 未満であること(帯であって面ではない)
    expect(att.cells.length).toBeLessThan(n * m * 0.05);
    for (const [i, j, v] of att.cells) {
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(n);
      expect(j).toBeGreaterThanOrEqual(0);
      expect(j).toBeLessThan(m);
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("目玉の判定がそのまま焼かれている(落ちたことを含めて)", () => {
    const v = att.verdict;
    expect(v.passed).toBe(false);
    expect(v.p_value).toBeGreaterThan(v.threshold);
    expect(v.difference).toBeGreaterThan(0); // 向きは正だが有意でない
    expect(v.attention.paragraph_agreement).toBeGreaterThan(
      v.diagonal.paragraph_agreement,
    );
    expect(v.p_display).toMatch(/^[<\d]/);
  });

  it("経路は帯の中にしか引かれていない", () => {
    const inBand = new Set(att.cells.map(([i, j]: number[]) => `${i},${j}`));
    for (const [i, j] of att.links) {
      expect(inBand.has(`${i},${j}`)).toBe(true);
    }
  });
});
