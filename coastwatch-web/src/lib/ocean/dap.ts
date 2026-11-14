/**
 * A small decoder for OPeNDAP DAP2 binary responses (`.dods`), as THREDDS (NOAA CO-OPS) and ERDDAP (NOAA
 * CoastWatch) serve them. A `.dods` body is the DDS text, a `Data:` line, then each declared variable in DDS
 * order, XDR-encoded (big-endian): an array is its element count twice, then the elements; a GRID is its array
 * followed by its map vectors; a scalar is the value alone. DAP2 sends Int16/UInt16 as 32-bit words and Byte
 * arrays padded to four bytes. Structures and sequences are not used by the products Neritic reads and are
 * refused rather than guessed at.
 */

export type DapType = "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Float32" | "Float64";
export type DapDim = { name: string; size: number };
export type DapVar = { name: string; type: DapType; dims: DapDim[] };
type DapDecl = { kind: "array"; v: DapVar } | { kind: "grid"; name: string; array: DapVar; maps: DapVar[] };

export type DapArray = DapVar & { data: Float64Array | Float32Array | Int32Array | Uint32Array | Uint8Array };

const TYPES: Record<string, DapType> = {
  byte: "Byte", int16: "Int16", uint16: "UInt16", int32: "Int32", uint32: "UInt32", float32: "Float32", float64: "Float64",
};

const DECL = /^\s*(\w+)\s+([\w.%-]+)((?:\s*\[[^\]]*\])*)\s*;\s*$/;

function parseVar(line: string): DapVar {
  const m = DECL.exec(line);
  if (!m) throw new Error(`DAP: cannot read declaration "${line.trim().slice(0, 80)}"`);
  const type = TYPES[m[1].toLowerCase()];
  if (!type) throw new Error(`DAP: unsupported type ${m[1]}`);
  const dims: DapDim[] = [];
  for (const d of m[3].matchAll(/\[\s*(?:([\w.-]+)\s*=\s*)?(\d+)\s*\]/g)) dims.push({ name: d[1] ?? "", size: Number(d[2]) });
  return { name: m[2], type, dims };
}

/** The declarations of a DDS, in order. */
export function parseDds(dds: string): DapDecl[] {
  const lines = dds.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!/^Dataset\s*\{$/i.test(lines[0] ?? "")) throw new Error("DAP: not a DDS (no Dataset header)");
  const out: DapDecl[] = [];
  let i = 1;
  for (; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith("}")) break; // end of Dataset
    if (/^grid\s*\{$/i.test(l)) {
      if (!/^array:$/i.test(lines[++i] ?? "")) throw new Error("DAP: GRID without ARRAY");
      const array = parseVar(lines[++i]);
      if (!/^maps:$/i.test(lines[++i] ?? "")) throw new Error("DAP: GRID without MAPS");
      const maps: DapVar[] = [];
      while (i + 1 < lines.length && !lines[i + 1].startsWith("}")) maps.push(parseVar(lines[++i]));
      const end = /^\}\s*([\w.%-]+)\s*;$/.exec(lines[++i] ?? "");
      if (!end) throw new Error("DAP: unterminated GRID");
      out.push({ kind: "grid", name: end[1], array, maps });
    } else if (/^(structure|sequence)\b/i.test(l)) {
      throw new Error("DAP: structures and sequences are not supported");
    } else {
      out.push({ kind: "array", v: parseVar(l) });
    }
  }
  if (i >= lines.length) throw new Error("DAP: unterminated Dataset");
  return out;
}

const SEP = [0x0a, 0x44, 0x61, 0x74, 0x61, 0x3a, 0x0a]; // "\nData:\n"

function findSeparator(b: Uint8Array): number {
  outer: for (let i = 0; i + SEP.length <= b.length && i < 1 << 20; i++) {
    for (let j = 0; j < SEP.length; j++) if (b[i + j] !== SEP[j]) continue outer;
    return i;
  }
  return -1;
}

const count = (v: DapVar) => v.dims.reduce((n, d) => n * d.size, 1);

function readVar(view: DataView, off: number, v: DapVar): { arr: DapArray; off: number } {
  const n = count(v);
  if (v.dims.length) {
    if (off + 8 > view.byteLength) throw new Error(`DAP: response ends before ${v.name}`);
    const n1 = view.getUint32(off), n2 = view.getUint32(off + 4);
    if (n1 !== n || n2 !== n) throw new Error(`DAP: ${v.name} declares ${n} values but the data says ${n1}`);
    off += 8;
  }
  const size = v.type === "Float64" ? 8 : v.type === "Byte" ? 1 : 4;
  const bytes = v.type === "Byte" ? Math.ceil(n / 4) * 4 : n * size;
  if (off + bytes > view.byteLength) throw new Error(`DAP: response ends inside ${v.name}`);
  let data: DapArray["data"];
  switch (v.type) {
    case "Float64": { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = view.getFloat64(off + i * 8); data = a; break; }
    case "Float32": { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = view.getFloat32(off + i * 4); data = a; break; }
    case "Int16": case "Int32": { const a = new Int32Array(n); for (let i = 0; i < n; i++) a[i] = view.getInt32(off + i * 4); data = a; break; }
    case "UInt16": case "UInt32": { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = view.getUint32(off + i * 4); data = a; break; }
    case "Byte": data = new Uint8Array(view.buffer.slice(view.byteOffset + off, view.byteOffset + off + n)); break;
  }
  return { arr: { ...v, data }, off: off + bytes };
}

/**
 * Decode a `.dods` body. Variables are keyed by name; a GRID's array is keyed by the grid's name and its maps by
 * `<grid>.<map>` (and by the map's own name when no top-level variable has it). An error body (THREDDS and
 * ERDDAP answer "Error { … }") or a truncated body throws.
 */
export function decodeDods(body: ArrayBuffer | Uint8Array): Map<string, DapArray> {
  const bytes = body instanceof Uint8Array ? body : new Uint8Array(body);
  const sep = findSeparator(bytes);
  const head = new TextDecoder().decode(bytes.subarray(0, sep < 0 ? Math.min(bytes.length, 400) : sep));
  if (sep < 0) {
    const err = /message\s*=\s*"([^"]*)"/.exec(head)?.[1];
    throw new Error(err ? `DAP error: ${err.slice(0, 160)}` : "DAP: no Data section");
  }
  const decls = parseDds(head);
  const view = new DataView(bytes.buffer, bytes.byteOffset + sep + SEP.length, bytes.length - sep - SEP.length);
  const out = new Map<string, DapArray>();
  let off = 0;
  for (const d of decls) {
    if (d.kind === "array") {
      const r = readVar(view, off, d.v);
      out.set(d.v.name, r.arr);
      off = r.off;
    } else {
      const a = readVar(view, off, d.array);
      out.set(d.name, { ...a.arr, name: d.name });
      off = a.off;
      for (const m of d.maps) {
        const r = readVar(view, off, m);
        out.set(`${d.name}.${m.name}`, r.arr);
        off = r.off;
      }
    }
  }
  for (const [k, v] of [...out]) {
    const dot = k.indexOf(".");
    if (dot > 0 && !out.has(k.slice(dot + 1))) out.set(k.slice(dot + 1), v);
  }
  return out;
}
