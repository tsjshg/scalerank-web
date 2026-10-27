// Synthetic demo table: 3 groups of 3 clusters each in 20 dimensions, with
// different spreads, so both the local (clusters) and the global (groups)
// structure matter.

import type { Table } from "./csv";

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function demoTable(n = 6000, d = 20): Table {
  const rand = mulberry32(7);
  const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const groups = ["A", "B", "C"];
  const spreads = [0.6, 1.0, 1.6];
  const centres: number[][] = [];
  const names: string[] = [];
  const sd: number[] = [];
  groups.forEach((g) => {
    const gc = Array.from({ length: d }, () => 12 * normal());
    for (let s = 0; s < 3; s++) {
      centres.push(gc.map((v) => v + 4 * normal()));
      names.push(`${g}${s + 1}`);
      sd.push(spreads[s]);
    }
  });
  const cols = Array.from({ length: d }, () => new Float32Array(n));
  const text: string[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % centres.length;
    for (let j = 0; j < d; j++) cols[j][i] = centres[c][j] + sd[c] * normal();
    text.push(names[c]);
  }
  return {
    n,
    columns: [
      ...cols.map((values, j) => ({ name: `x${j + 1}`, numeric: true, values, text: null, distinct: 1000 })),
      { name: "cluster", numeric: false, values: new Float32Array(n).fill(Number.NaN), text, distinct: names.length },
    ],
  };
}
