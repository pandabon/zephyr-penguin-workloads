"""One SmolVLA action-expert layer as a standalone torch module (denoise phase, KV cache given).

Faithful re-implementation of the expert path of lerobot's SmolVLA, for export with ExecuTorch:
  lerobot (huggingface/lerobot @ 8a3d6403, v0.5.2, Apache-2.0),
  src/lerobot/policies/smolvla/smolvlm_with_expert.py:
    apply_rope()                       lines 39-62
    forward_attn_layer()               lines 209-284  (self-attn layers: even index)
    forward_cross_attn_layer()         lines 286-399  (cross-attn layers: odd index, expert branch)
    forward() residual/MLP part        lines 470-496
    eager_attention_forward()          lines 516-561
  RMSNorm and the SwiGLU MLP are those of HF transformers' Llama (SmolVLM2's text model).
Differences from lerobot, on purpose (export only, the math is the same):
  * positions are fixed, so the RoPE sin/cos tables are precomputed in __init__: cross-attn layers
    use 0..L-1 (lerobot subtracts the min, line 378), self-attn layers use the suffix positions
    P..P+L-1 that lerobot passes unchanged (pos_offset = number of valid prefix tokens);
  * the attention mask is all-true (no padded prefix tokens), so torch.where(mask, ...) is dropped
    (this single-layer module only; a full denoise-step export would take the prefix pad mask and the
    RoPE sin/cos of the self-attn layers as runtime inputs);
  * the `res = empty_like(x); res[..., :d] = ...` RoPE writes become a torch.cat (same values).

Weights come straight from the lerobot/smolvla_base checkpoint (model.safetensors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX = "model.vlm_with_expert.lm_expert.layers.{}."
HEADS, KV_HEADS, HEAD_DIM = 15, 5, 64          # SmolVLM2 text config (expert keeps heads/head_dim)
RMS_EPS = 1e-5                                 # SmolVLM2 text config rms_norm_eps
ROPE_MAX_WAVELENGTH = 10_000                   # lerobot apply_rope default (not the HF rope_theta)


def rope_tables(seq_len, offset=0, head_dim=HEAD_DIM, max_wavelength=ROPE_MAX_WAVELENGTH):
    """sin/cos [L, 1, D/2] for positions offset..offset+L-1, as lerobot apply_rope() computes them."""
    d_half = head_dim // 2
    freq_exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32)
    timescale = max_wavelength ** freq_exponents
    radians = (offset + torch.arange(seq_len, dtype=torch.float32))[:, None] / timescale[None, :]
    return torch.sin(radians)[:, None, :], torch.cos(radians)[:, None, :]


class RMSNorm(nn.Module):
    """transformers LlamaRMSNorm."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + RMS_EPS))


class ExpertLayer(nn.Module):
    """Expert layer `idx` for `seq_len` action tokens attending to a cached prefix.

    forward(x, prefix_k, prefix_v):
      x         [1, L, 720]   expert hidden states
      prefix_k  [1, P, 320]   cached keys of the prefix (VLM layer output, RoPE already applied)
      prefix_v  [1, P, 320]   cached values of the prefix
    Self-attn layer (even idx): keys/values = cat(prefix, expert's own k/v), RoPE on q and expert k.
    Cross-attn layer (odd idx): keys/values = k_proj/v_proj(prefix) (320 -> 320), RoPE on q only.
    """

    def __init__(self, idx, seq_len, pos_offset=0, hidden=720, inter=2048, kv_in=None):
        super().__init__()
        self.idx, self.cross = idx, idx % 2 == 1
        kv_in = kv_in or (KV_HEADS * HEAD_DIM if self.cross else hidden)
        self.input_layernorm = RMSNorm(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(kv_in, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(kv_in, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        sin, cos = rope_tables(seq_len, 0 if self.cross else pos_offset)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def rope(self, x, sin=None, cos=None):  # x [1, L, H, D]; sin/cos [L,1,D/2] given at run time or the baked tables
        sin = self.sin if sin is None else sin
        cos = self.cos if cos is None else cos
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def attention(self, q, k, v):  # q [1,L,H,D], k/v [1,S,KVH,D] -> [1, L, H*D]
        groups = HEADS // KV_HEADS
        k = k.repeat_interleave(groups, dim=2)   # same layout as lerobot's expand+reshape
        v = v.repeat_interleave(groups, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = torch.matmul(q, k.transpose(2, 3)) * HEAD_DIM ** -0.5
        probs = F.softmax(att, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], HEADS * HEAD_DIM)

    def forward(self, x, prefix_k, prefix_v):
        L = x.shape[1]
        h = self.input_layernorm(x)
        q = self.rope(self.q_proj(h).view(1, L, HEADS, HEAD_DIM))
        if self.cross:
            P = prefix_k.shape[1]
            k = self.k_proj(prefix_k).view(1, P, KV_HEADS, HEAD_DIM)
            v = self.v_proj(prefix_v).view(1, P, KV_HEADS, HEAD_DIM)
        else:
            P = prefix_k.shape[1]
            ek = self.rope(self.k_proj(h).view(1, L, KV_HEADS, HEAD_DIM))
            ev = self.v_proj(h).view(1, L, KV_HEADS, HEAD_DIM)
            k = torch.cat([prefix_k.view(1, P, KV_HEADS, HEAD_DIM), ek], dim=1)
            v = torch.cat([prefix_v.view(1, P, KV_HEADS, HEAD_DIM), ev], dim=1)
        out = self.o_proj(self.attention(q, k, v)) + x
        h = self.post_attention_layernorm(out)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + out


def load_expert_layer(safetensors_path, idx, seq_len=50, pos_offset=0, dtype=torch.float32):
    from safetensors import safe_open
    m = ExpertLayer(idx, seq_len, pos_offset)
    names = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "self_attn.q_proj.weight": "q_proj.weight", "self_attn.k_proj.weight": "k_proj.weight",
        "self_attn.v_proj.weight": "v_proj.weight", "self_attn.o_proj.weight": "o_proj.weight",
        "mlp.gate_proj.weight": "gate_proj.weight", "mlp.up_proj.weight": "up_proj.weight",
        "mlp.down_proj.weight": "down_proj.weight",
    }
    sd = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for ck, mk in names.items():
            sd[mk] = f.get_tensor(PREFIX.format(idx) + ck).to(torch.float32)
    m.load_state_dict(sd)
    return m.to(dtype).eval()


class ExpertLayerWL5(ExpertLayer):
    """The same layer with our mixed-precision policy (bf16 only at matmul inputs), written as explicit casts.

    The mixed-precision policy (chosen for the Penguin bring-up):
      * weights bf16, except weights the checkpoint stores in f32 (the cross-attn layers' k_proj / v_proj,
        320x320): those stay f32, and so does their input (the prefix K/V, which enter as f32);
      * activations are cast to bf16 ONLY at the inputs of matmuls: q/k/v/o linears, MLP gate/up/down, and
        the two attention batch-matmuls (QK^T, P*V);
      * every matmul is followed by `.to(float32)`: the intent is "bf16 x bf16 -> f32 accumulate -> f32 output",
        which is what XNNPACK's bf16_f32 gemm computes;
      * everything else stays f32: residual stream, RMSNorm statistics and weight multiply, RoPE, the attention
        scale, softmax, SiLU and gate*up.
    Known gap (reported, not fixed): PyTorch's bf16 matmul returns a bf16 tensor, so in eager mode the matmul
    result is rounded to bf16 once before the `.to(float32)`; the intended f32 output would skip that rounding.
    In the exported graph the same happens in a different form (XNNPACK computes f32, ExecuTorch converts it to
    bf16 because the graph says bf16, then the cast widens it again). Fixing it needs a graph pass that fuses
    "bf16 matmul -> to(float32)" into one matmul node with f32 output (see the export script's report).
    """

    BF16, F32 = torch.bfloat16, torch.float32

    def to_wl5(self):
        bf16_mods = [self.q_proj, self.o_proj, self.gate_proj, self.up_proj, self.down_proj]
        if not self.cross:               # self-attn layers store k_proj / v_proj in bf16; cross-attn keeps f32
            bf16_mods += [self.k_proj, self.v_proj]
        for m in bf16_mods:
            m.to(self.BF16)
        return self

    def mm(self, lin, x_bf16):
        """bf16 linear, widened back to f32 right after (x_bf16 is already cast by the caller)."""
        return lin(x_bf16).to(self.F32)

    def attention(self, q, k, v):  # q [1,L,H,D], k/v [1,S,KVH,D], all f32 -> [1, L, H*D] f32
        groups = HEADS // KV_HEADS
        # Cast k and v BEFORE the GQA repeat: the same values, but 3x fewer elements to convert.
        k = k.to(self.BF16).repeat_interleave(groups, dim=2)
        v = v.to(self.BF16).repeat_interleave(groups, dim=2)
        q, k, v = q.to(self.BF16).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = torch.matmul(q, k.transpose(2, 3)).to(self.F32) * HEAD_DIM ** -0.5   # QK^T, scale in f32
        probs = F.softmax(att, dim=-1)                                              # f32
        out = torch.matmul(probs.to(self.BF16), v).to(self.F32).transpose(1, 2)     # P*V
        return out.reshape(out.shape[0], out.shape[1], HEADS * HEAD_DIM)

    def forward(self, x, prefix_k, prefix_v):
        L = x.shape[1]
        h = self.input_layernorm(x)                       # f32 norm (stats and weight multiply)
        hb = h.to(self.BF16)                              # one cast, shared by q (and k, v in self-attn layers)
        q = self.rope(self.mm(self.q_proj, hb).view(1, L, HEADS, HEAD_DIM))
        P = prefix_k.shape[1]
        if self.cross:
            k = self.k_proj(prefix_k).view(1, P, KV_HEADS, HEAD_DIM)      # f32 weights, f32 input (exception of the mixed-precision policy)
            v = self.v_proj(prefix_v).view(1, P, KV_HEADS, HEAD_DIM)
        else:
            ek = self.rope(self.mm(self.k_proj, hb).view(1, L, KV_HEADS, HEAD_DIM))
            ev = self.mm(self.v_proj, hb).view(1, L, KV_HEADS, HEAD_DIM)
            k = torch.cat([prefix_k.view(1, P, KV_HEADS, HEAD_DIM), ek], dim=1)
            v = torch.cat([prefix_v.view(1, P, KV_HEADS, HEAD_DIM), ev], dim=1)
        ab = self.attention(q, k, v).to(self.BF16)        # o_proj input
        out = self.mm(self.o_proj, ab) + x                # residual add in f32
        h = self.post_attention_layernorm(out)
        hb = h.to(self.BF16)                              # shared by gate and up
        gated = F.silu(self.mm(self.gate_proj, hb)) * self.mm(self.up_proj, hb)   # SiLU and gate*up in f32
        return self.mm(self.down_proj, gated.to(self.BF16)) + out


def load_expert_layer_wl5(safetensors_path, idx, seq_len=50, pos_offset=0):
    """Load the layer in f32 (exact: the checkpoint is bf16/f32) and apply the mixed-precision policy's weight dtypes."""
    m = load_expert_layer(safetensors_path, idx, seq_len, pos_offset, dtype=torch.float32)
    wl5 = ExpertLayerWL5(idx, seq_len, pos_offset)
    wl5.load_state_dict(m.state_dict())
    return wl5.to_wl5().eval()


class ExpertLayerLerobotBF16(ExpertLayer):
    """The layer with lerobot's OWN dtype handling, every cast mirrored from lerobot v0.5.2 as loaded.

    Source: huggingface/lerobot @ 8a3d6403 (v0.5.2, Apache-2.0), src/lerobot/policies/smolvla/
      smolvlm_with_expert.py  apply_rope 39-62, forward_attn_layer 209-284, forward_cross_attn_layer 286-399,
                              forward() residual + MLP 470-496, eager_attention_forward 516-561
      modeling_smolvla.py     embed_suffix 736-775 (what the expert's first layer is fed)
    RMSNorm / SwiGLU MLP: HF transformers 5.3.0 models/llama/modeling_llama.py, LlamaRMSNorm 53-67, LlamaMLP 171-184.
    "As loaded" = the dtypes stored in lerobot/smolvla_base model.safetensors: expert weights bf16, except the
    cross-attn k_proj / v_proj (320x320, odd layers) which are f32; action_in_proj / action_time_mlp_* are f32.
    (Caveat: if lerobot instead copied the checkpoint into f32 parameters, everything would be f32; we follow the
    checkpoint dtypes, as the bring-up plan says.)

    dtype map (op -> dtype; "lerobot" = file:line above):
      expert input x   layer 0: f32  (embed_suffix: action_time_mlp_out is an f32 Linear, lines 738-757)
                       layer >=1: bf16 (output of the previous layer's `out_emb += ...`, line 494)
      prefix K / V     bf16 (VLM layer's apply_rope returns `res.to(dtype)` bf16, line 62; V is a bf16 Linear output);
                       the cache holds them as they are (lines 270-284 / 340-346)
      input_layernorm  statistics in f32; `weight * h.to(input_dtype)`: bf16 x bf16 -> bf16 (layer 1),
                       bf16 weight * f32 x -> f32 (layer 0)   (LlamaRMSNorm 62-67)
      q / (self-attn) k / v proj input   `.to(q_proj.weight.dtype)` = bf16 (lines 234, 360); outputs bf16
      cross-attn K/V projection          `key_states.to(k_proj.weight.dtype)` = f32 (lines 363-373): the bf16 prefix
                                         K/V are widened to f32, k_proj / v_proj run in f32 -> f32 K and V
      RoPE (apply_rope)  x.to(f32), f32 sin/cos, f32 math, `.to(dtype)` back to the input's dtype (lines 45-62):
                         q -> bf16; self-attn expert k -> bf16; cross-attn K gets NO rope (stays f32)
      GQA expand         expand + reshape in the tensor's own dtype (lines 520-534)
      QK^T               `query.to(f32)`, `key.to(f32)`, f32 matmul (lines 540-547): an f32 x f32 matmul (the values are
                         bf16-representable in self-attn layers, but cross-attn K is a true f32)
      scale, softmax     f32 (lines 548-553; the where(mask) is dropped: mask all-true)
      probs              `.to(value_states.dtype)` (line 553): bf16 in self-attn layers (bf16 V), f32 in cross-attn
                         layers (f32 V)
      PV                 in the V dtype: bf16 x bf16 -> bf16 (self-attn), f32 x f32 -> f32 (cross-attn) (line 555)
      o_proj input       `att_output.to(o_proj.weight.dtype)` = bf16 (lines 481-482); o_proj output bf16
      residual 1         `out_emb += hidden_states` (line 487): in-place on the bf16 out_emb -> bf16 result (the f32
                         x of layer 0 is added in f32 and rounded once)
      post_attention_layernorm / MLP   all bf16 (bf16 in, f32 stats inside the norm, bf16 out); gate, up, silu (bf16 op),
                         gate*up (bf16 mul), down all bf16 (LlamaMLP 182-184)
      residual 2         `out_emb += after_first_residual` bf16 + bf16 -> bf16 (line 494)
      output             bf16
    Same export-only differences as ExpertLayer (precomputed RoPE tables, all-true mask, cat instead of the
    `res[..., :d] = ...` writes; the same values).
    """

    BF16, F32 = torch.bfloat16, torch.float32

    def __init__(self, idx, seq_len, pos_offset=0, **kw):
        super().__init__(idx, seq_len, pos_offset, **kw)
        self.x_dtype = self.F32 if idx == 0 else self.BF16     # see the dtype map: what the layer is fed

    def to_lerobot_bf16(self):
        for m in [self.q_proj, self.o_proj, self.gate_proj, self.up_proj, self.down_proj,
                  self.input_layernorm, self.post_attention_layernorm]:
            m.to(self.BF16)                      # checkpoint: bf16
        if not self.cross:                       # cross-attn k_proj / v_proj are f32 in the checkpoint
            self.k_proj.to(self.BF16)
            self.v_proj.to(self.BF16)
        return self

    def norm(self, n, x):
        """LlamaRMSNorm (modeling_llama.py:62-67): f32 statistics, `weight * h.to(input_dtype)`."""
        input_dtype = x.dtype
        h = x.to(self.F32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + RMS_EPS)
        return n.weight * h.to(input_dtype)      # bf16 weight * f32 -> f32 (layer 0 input); bf16 * bf16 -> bf16

    def rope(self, x, sin=None, cos=None):
        """apply_rope (smolvlm_with_expert.py:39-62): f32 math, result cast back to x's dtype."""
        dtype = x.dtype
        x = x.to(self.F32)
        return super().rope(x, sin, cos).to(dtype)

    def attention(self, q, k, v):
        """eager_attention_forward (516-561). q [1,L,H,D] bf16, k/v [1,S,KVH,D] bf16 (self) or f32 (cross)."""
        groups = HEADS // KV_HEADS
        k = k.repeat_interleave(groups, dim=2)   # expand + reshape (520-534), tensor's own dtype
        v = v.repeat_interleave(groups, dim=2)
        q = q.to(self.F32).transpose(1, 2)       # 540, 543
        k = k.to(self.F32).transpose(1, 2)       # 541, 544
        att = torch.matmul(q, k.transpose(2, 3))  # 546: f32 x f32
        att = att * HEAD_DIM ** -0.5             # 547
        probs = F.softmax(att, dim=-1)           # 551 (f32)
        probs = probs.to(v.dtype)                # 553: V dtype (bf16 self-attn, f32 cross-attn)
        out = torch.matmul(probs, v.transpose(1, 2))   # 555
        out = out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], HEADS * HEAD_DIM)   # 558

    def forward(self, x, prefix_k, prefix_v):
        L, P = x.shape[1], prefix_k.shape[1]
        h = self.norm(self.input_layernorm, x)
        h = h.to(self.q_proj.weight.dtype)                            # lines 234 / 360
        q = self.q_proj(h).view(1, L, HEADS, HEAD_DIM)
        if self.cross:
            _k = prefix_k.to(self.k_proj.weight.dtype).view(1, P, -1)   # lines 363-365: bf16 cache -> f32
            k = self.k_proj(_k).view(1, P, KV_HEADS, HEAD_DIM)          # f32 Linear, f32 out, no RoPE
            _v = prefix_v.to(self.v_proj.weight.dtype).view(1, P, -1)   # lines 370-372
            v = self.v_proj(_v).view(1, P, KV_HEADS, HEAD_DIM)
            q = self.rope(q)                                            # line 387 (positions from 0)
        else:
            ek = self.k_proj(h).view(1, L, KV_HEADS, HEAD_DIM)
            ev = self.v_proj(h).view(1, L, KV_HEADS, HEAD_DIM)
            q, ek = self.rope(q), self.rope(ek)                         # lines 262-263
            k = torch.cat([prefix_k.view(1, P, KV_HEADS, HEAD_DIM), ek], dim=1)   # lines 280-281
            v = torch.cat([prefix_v.view(1, P, KV_HEADS, HEAD_DIM), ev], dim=1)
        att = self.attention(q, k, v)
        att = att.to(self.o_proj.weight.dtype)                          # lines 481-482
        out = self.o_proj(att)
        out = torch.add(out, x).to(out.dtype)       # 487 `out_emb += hidden_states`: in-place keeps out_emb's bf16
        h = self.norm(self.post_attention_layernorm, out)               # 490
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))  # 491 LlamaMLP: all bf16
        return h + out                                                  # 494


def load_expert_layer_lerobot_bf16(safetensors_path, idx, seq_len=50, pos_offset=0):
    """Load the layer from the checkpoint and give it lerobot's dtypes (bf16 weights, f32 cross-attn k/v proj)."""
    m = load_expert_layer(safetensors_path, idx, seq_len, pos_offset, dtype=torch.float32)  # exact upcast
    lb = ExpertLayerLerobotBF16(idx, seq_len, pos_offset)
    lb.load_state_dict(m.state_dict())
    return lb.to_lerobot_bf16().eval()
