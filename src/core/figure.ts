// 図の座標まわり。**ラベルの位置も内容も、図を描いたのと同じデータから導く**(HC-045)。
// 決め打ちの座標を置くと、データが動いたときに図だけが嘘になる。

export type Box = { left: number; top: number; right: number; bottom: number };

/** 図の外枠。余白は軸ラベルが**外に**出るぶんを見込んで取る(HC-159)。 */
export const PLOT: Box = { left: 56, top: 18, right: 16, bottom: 44 };

export type Scale = (v: number) => number;

/**
 * `[0, n)` を `[lo, hi]` に写す。**n が 1 のときも落ちない**ようにする ——
 * 一文しかない章は無いが、頁分けの端では起こりうる。
 */
export function linear(n: number, lo: number, hi: number): Scale {
  if (n <= 1) return () => (lo + hi) / 2;
  const span = hi - lo;
  return (v) => lo + (v / (n - 1)) * span;
}

/**
 * 段落の切れ目(その段落の最初の文の番号)。
 * `paragraph` は文ごとの段落番号で、範囲 `[start, end)` を見る。
 */
export function paragraphStarts(
  paragraph: number[],
  start: number,
  end: number,
): number[] {
  const out: number[] = [];
  let current: number | null = null;
  for (let i = start; i < end; i += 1) {
    if (paragraph[i] !== current) {
      current = paragraph[i];
      out.push(i - start);
    }
  }
  return out;
}

/** 目盛りの値。**軸の長さから本数を決める** —— 決め打ちだと詰まるか間延びする。 */
export function ticks(n: number, approx: number): number[] {
  if (n <= 1) return [0];
  const raw = Math.max(1, Math.round(n / Math.max(approx, 1)));
  const nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find((s) => s >= raw)
    ?? Math.ceil(raw / 100) * 100;
  const out: number[] = [];
  for (let v = 0; v < n; v += nice) out.push(v);
  if (out[out.length - 1] !== n - 1) out.push(n - 1);
  return out;
}

/**
 * 対応の列を、単調な折れ線の点列に畳む。
 * 同じ src に複数の dst があるときは**中央**を取る —— 端を取ると経路が
 * 実際より上か下に寄って見える。
 */
export function pathPoints(links: [number, number][]): [number, number][] {
  const byI = new Map<number, number[]>();
  for (const [i, j] of links) {
    const cur = byI.get(i);
    if (cur) cur.push(j);
    else byI.set(i, [j]);
  }
  return [...byI.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([i, js]) => {
      const sorted = [...js].sort((x, y) => x - y);
      return [i, sorted[(sorted.length - 1) >> 1]] as [number, number];
    });
}

/** 点列を SVG の折れ線に。空なら空文字(`<path d="">` は描かれない)。 */
export function polyline(
  points: [number, number][],
  sx: Scale,
  sy: Scale,
): string {
  if (!points.length) return "";
  return points.map((p, k) => `${k ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
}
