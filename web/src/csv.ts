// Minimal delimited-text reader: detects the delimiter (comma, tab,
// semicolon) and a header row, handles quoted fields, and classifies each
// column as numeric or categorical.

export interface Column {
  name: string;
  numeric: boolean;
  /** numeric value per row (NaN when missing or not a number) */
  values: Float32Array;
  /** raw text per row, kept for non-numeric columns only */
  text: string[] | null;
  /** distinct non-empty values, counted up to 1000 */
  distinct: number;
}

export interface Table {
  n: number;
  columns: Column[];
}

const MISSING = new Set(["", "na", "nan", "null", "none", "?", "-"]);

function detectDelimiter(sample: string): string {
  const lines = sample.split(/\r?\n/).slice(0, 20).filter((l) => l.length > 0);
  let best = ",";
  let bestScore = -1;
  for (const c of [",", "\t", ";"]) {
    const counts = lines.map((l) => l.split(c).length - 1);
    const min = Math.min(...counts);
    // consistent and frequent
    const score = min > 0 && counts.every((k) => k === counts[0]) ? min + 1000 : min;
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

/** Splits text into rows of fields (RFC 4180 quoting). */
function* records(text: string, delim: string): Generator<string[]> {
  let field = "";
  let row: string[] = [];
  let i = 0;
  let quoted = false;
  const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        quoted = false;
      } else {
        field += ch;
      }
      i++;
      continue;
    }
    if (ch === '"' && field.length === 0) {
      quoted = true;
    } else if (ch === delim) {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      row.push(field);
      field = "";
      if (ch === "\r" && text[i + 1] === "\n") i++;
      if (row.length > 1 || row[0] !== "") yield row;
      row = [];
    } else {
      field += ch;
    }
    i++;
  }
  row.push(field);
  if (row.length > 1 || row[0] !== "") yield row;
}

function toNumber(s: string): number {
  const t = s.trim();
  if (MISSING.has(t.toLowerCase())) return NaN;
  const v = Number(t);
  return t.length > 0 && Number.isFinite(v) ? v : Number.NaN;
}

function isNumberLike(s: string): boolean {
  const t = s.trim();
  return MISSING.has(t.toLowerCase()) || Number.isFinite(Number(t));
}

export function parseTable(text: string): Table {
  const delim = detectDelimiter(text.slice(0, 64 * 1024));
  const rows = [...records(text, delim)];
  if (rows.length < 2) throw new Error("The file has fewer than two rows.");
  const width = Math.max(...rows.slice(0, 50).map((r) => r.length));
  // header: the first row has a non-numeric entry where the second row is numeric,
  // or the first row is entirely non-numeric
  const first = rows[0];
  const second = rows[1];
  const header =
    first.every((f) => !isNumberLike(f) || f.trim() === "") ||
    first.some((f, j) => !isNumberLike(f) && j < second.length && isNumberLike(second[j]) && second[j].trim() !== "");
  const names = header ? first.map((f, j) => f.trim() || `column ${j + 1}`) : [];
  const body = header ? rows.slice(1) : rows;
  const n = body.length;
  const columns: Column[] = [];
  for (let j = 0; j < width; j++) {
    const values = new Float32Array(n);
    let nonNumeric = 0;
    const seen = new Set<string>();
    for (let i = 0; i < n; i++) {
      const raw = body[i][j] ?? "";
      const v = toNumber(raw);
      values[i] = v;
      const t = raw.trim();
      if (Number.isNaN(v) && !MISSING.has(t.toLowerCase())) nonNumeric++;
      if (seen.size < 1000 && t !== "") seen.add(t);
    }
    const numeric = nonNumeric === 0;
    columns.push({
      name: names[j] ?? `column ${j + 1}`,
      numeric,
      values,
      text: numeric ? null : body.map((r) => (r[j] ?? "").trim()),
      distinct: seen.size,
    });
  }
  return { n, columns };
}

/** A sensible default label column: categorical with 2..100 distinct values. */
export function guessLabelColumn(t: Table): number {
  const ok = (c: Column) => c.distinct >= 2 && c.distinct <= Math.min(100, t.n / 2);
  const cat = t.columns.findIndex((c) => !c.numeric && ok(c));
  return cat;
}

/** Category index per row (u32::MAX for missing) and the category names by frequency. */
export function encodeLabels(c: Column, n: number): { codes: Uint32Array; names: string[] } {
  const text = c.text ?? Array.from(c.values, (v) => (Number.isNaN(v) ? "" : String(v)));
  const counts = new Map<string, number>();
  for (const t of text) if (t !== "") counts.set(t, (counts.get(t) ?? 0) + 1);
  const names = [...counts.entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0]);
  const index = new Map(names.map((s, k) => [s, k]));
  const codes = new Uint32Array(n);
  for (let i = 0; i < n; i++) codes[i] = text[i] === "" ? 0xffffffff : index.get(text[i])!;
  return { codes, names };
}
