// Black Whole — browser inference for the trained char-level model.
//
// The weights are int8-quantised (one scale per tensor) and shipped as one JSON
// file, so the model runs on the reader's own device with no server, no API key
// and no cost. Forward pass is a plain GPT with a KV cache; 80 tokens of output
// takes a couple of seconds on a phone.
//
// This is deliberately small code on purpose: everything here is the model that
// was actually trained, not a stand-in.

const BW = (() => {
  let W = null;        // dequantised Float32Arrays by tensor name
  let cfg = null;
  let stoi = null, itos = null;

  const GELU = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function dequant(t) {
    const ints = b64ToBytes(t.data);
    const f = new Float32Array(ints.length);
    const s = t.scale;
    for (let i = 0; i < ints.length; i++) {
      const v = ints[i] < 128 ? ints[i] : ints[i] - 256;
      f[i] = v * s;
    }
    return f;
  }

  async function load(url) {
    const res = await fetch(url);
    const j = await res.json();
    cfg = j.config; stoi = j.stoi; itos = j.itos;
    W = {};
    for (const [name, t] of Object.entries(j.tensors)) W[name] = { v: dequant(t), shape: t.shape };
    return cfg;
  }

  // row-major matmul: y[o] = sum_i x[i] * M[o*inDim + i]
  function matvec(x, M, inDim, outDim) {
    const y = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let acc = 0;
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) acc += x[i] * M[base + i];
      y[o] = acc;
    }
    return y;
  }

  function layerNorm(x, g, b) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let varr = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mean; varr += d * d; }
    varr /= n;
    const inv = 1 / Math.sqrt(varr + 1e-5);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
    return y;
  }

  function softmaxInPlace(a) {
    let m = -Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
    let s = 0;
    for (let i = 0; i < a.length; i++) { a[i] = Math.exp(a[i] - m); s += a[i]; }
    for (let i = 0; i < a.length; i++) a[i] /= s;
    return a;
  }

  // One step with a KV cache. state = {keys: [L]Float32Array, vals: [...], len}
  function step(tokenId, pos, state) {
    const { D, L, H } = cfg;
    const dh = D / H;
    let x = new Float32Array(D);
    const tokW = W['tok.weight'].v, posW = W['pos.weight'].v;
    for (let i = 0; i < D; i++) x[i] = tokW[tokenId * D + i] + posW[pos * D + i];

    for (let l = 0; l < L; l++) {
      const ln1g = W[`blocks.${l}.ln1.weight`].v, ln1b = W[`blocks.${l}.ln1.bias`].v;
      const h1 = layerNorm(x, ln1g, ln1b);

      const inW = W[`blocks.${l}.attn.in_proj_weight`].v;
      const inB = W[`blocks.${l}.attn.in_proj_bias`].v;
      const q = new Float32Array(D), k = new Float32Array(D), v = new Float32Array(D);
      for (let o = 0; o < D; o++) {
        let a = inB[o], b2 = inB[D + o], c2 = inB[2 * D + o];
        const r0 = o * D, r1 = (D + o) * D, r2 = (2 * D + o) * D;
        for (let i = 0; i < D; i++) {
          const hi = h1[i];
          a += hi * inW[r0 + i];
          b2 += hi * inW[r1 + i];
          c2 += hi * inW[r2 + i];
        }
        q[o] = a; k[o] = b2; v[o] = c2;
      }

      state.keys[l].push(k); state.vals[l].push(v);
      const nCtx = state.keys[l].length;
      const ctx = new Float32Array(D);

      for (let h = 0; h < H; h++) {
        const off = h * dh;
        const scores = new Float32Array(nCtx);
        for (let t = 0; t < nCtx; t++) {
          const kt = state.keys[l][t];
          let s = 0;
          for (let d = 0; d < dh; d++) s += q[off + d] * kt[off + d];
          scores[t] = s / Math.sqrt(dh);
        }
        softmaxInPlace(scores);
        for (let d = 0; d < dh; d++) {
          let acc = 0;
          for (let t = 0; t < nCtx; t++) acc += scores[t] * state.vals[l][t][off + d];
          ctx[off + d] = acc;
        }
      }

      const oW = W[`blocks.${l}.attn.out_proj.weight`].v;
      const oB = W[`blocks.${l}.attn.out_proj.bias`].v;
      const proj = matvec(ctx, oW, D, D);
      for (let i = 0; i < D; i++) x[i] += proj[i] + oB[i];

      const ln2g = W[`blocks.${l}.ln2.weight`].v, ln2b = W[`blocks.${l}.ln2.bias`].v;
      const h2 = layerNorm(x, ln2g, ln2b);
      const fcW = W[`blocks.${l}.mlp.0.weight`].v, fcB = W[`blocks.${l}.mlp.0.bias`].v;
      const pW = W[`blocks.${l}.mlp.2.weight`].v, pB = W[`blocks.${l}.mlp.2.bias`].v;
      const hid = matvec(h2, fcW, D, 4 * D);
      for (let i = 0; i < hid.length; i++) hid[i] = GELU(hid[i]) + fcB[i];
      const back = matvec(hid, pW, 4 * D, D);
      for (let i = 0; i < D; i++) x[i] += back[i] + pB[i];
    }

    x = layerNorm(x, W['ln_f.weight'].v, W['ln_f.bias'].v);
    const logits = matvec(x, W['head.weight'].v, D, cfg.V);
    return logits;
  }

  function newState() {
    return { keys: Array.from({ length: cfg.L }, () => []), vals: Array.from({ length: cfg.L }, () => []), pos: 0 };
  }

  function sample(logits, temp) {
    const t = Math.max(0.05, temp);
    let m = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
    const p = new Float32Array(logits.length);
    let s = 0;
    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - m) / t); s += p[i]; }
    let r = Math.random() * s;
    for (let i = 0; i < logits.length; i++) { r -= p[i]; if (r <= 0) return i; }
    return logits.length - 1;
  }

  async function generate(prompt, opts = {}) {
    const maxNew = Math.min(opts.maxNew ?? 80, 160);
    const temp = opts.temperature ?? 0.75;
    const state = newState();
    let text = '';
    let last = null;
    const ids = [];
    for (const ch of prompt) if (ch in stoi) ids.push(stoi[ch]);
    if (!ids.length) ids.push(stoi['\n'] ?? 0);

    for (const id of ids) {
      if (state.pos >= cfg.BLOCK) break;
      last = step(id, state.pos, state); state.pos++;
    }
    for (let i = 0; i < maxNew; i++) {
      if (state.pos >= cfg.BLOCK) break;
      const next = sample(last, temp);
      text += itos[next] ?? '';
      last = step(next, state.pos, state); state.pos++;
      if (opts.onToken) opts.onToken(text);
    }
    return text;
  }

  return { load, generate, get config() { return cfg; } };
})();
