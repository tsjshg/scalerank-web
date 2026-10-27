// Point colours: a categorical palette in fixed order (Okabe-Ito, extended
// with three of Paul Tol's), grey for the rest; a sequential ramp for values
// in [0, 1].

export const CATEGORICAL = [
  "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
  "#D55E00", "#F0E442", "#882255", "#44AA99", "#999933",
];
export const OTHER = "#9a9a9a";

// light to dark blue-green (viridis-like, reversed so high = dark)
const RAMP = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"];

export function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function categoricalColors(codes: Uint32Array, alpha: number): Uint8Array {
  const pal = CATEGORICAL.map(hexToRgb);
  const other = hexToRgb(OTHER);
  const out = new Uint8Array(codes.length * 4);
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i] < pal.length ? pal[codes[i]] : other;
    out.set([c[0], c[1], c[2], alpha], 4 * i);
  }
  return out;
}

export function uniformColors(n: number, hex: string, alpha: number): Uint8Array {
  const c = hexToRgb(hex);
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) out.set([c[0], c[1], c[2], alpha], 4 * i);
  return out;
}

export function rampColor(t: number): [number, number, number] {
  const ramp = RAMP.map(hexToRgb);
  const x = Math.min(1, Math.max(0, t)) * (ramp.length - 1);
  const k = Math.min(ramp.length - 2, Math.floor(x));
  const f = x - k;
  return [0, 1, 2].map((c) => Math.round(ramp[k][c] + f * (ramp[k + 1][c] - ramp[k][c]))) as [number, number, number];
}

export function sequentialColors(values: Float32Array, alpha: number): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    const c = rampColor(values[i]);
    out.set([c[0], c[1], c[2], alpha], 4 * i);
  }
  return out;
}

export const RAMP_CSS = `linear-gradient(to right, ${RAMP.join(", ")})`;
