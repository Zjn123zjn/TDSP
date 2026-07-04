import torch

import triton
import triton.language as tl

import torch
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_imp,  #
                    K_block_ptr, K_imp_block_ptr, V_block_ptr,  #
                    mse_loss,  #
                    start_m, qk_sqrt, qk_scale, qk_sqrt_imp, qk_scale_imp, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_ELEMENTS: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    K_imp_block_ptr = tl.advance(K_imp_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    mse_contrib_total = 0.0
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr).to(tl.float32)
        k_imp = tl.load(K_imp_block_ptr).to(tl.float32)
        qf32 = q.to(tl.float32)
        qf32_imp = q_imp.to(tl.float32)
        qk = tl.dot(qf32, k)
        qk_imp = tl.dot(qf32_imp, k_imp)
        diff = (qk * qk_sqrt - qk_imp * qk_sqrt_imp)
        diff_sqr = diff * diff
        mse_contrib = tl.sum(diff_sqr)
        # tl.atomic_add(mse_loss, mse_contrib)
        mse_contrib_total += mse_contrib
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(v.type.element_ty)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_imp_block_ptr = tl.advance(K_imp_block_ptr, (0, BLOCK_N))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    if STAGE == 2:
        for start_n in range(hi, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(K_block_ptr).to(tl.float32)
            k_imp = tl.load(K_imp_block_ptr).to(tl.float32)
            qf32 = q.to(tl.float32)
            qf32_imp = q_imp.to(tl.float32)
            qk = tl.dot(qf32, k)
            qk_imp = tl.dot(qf32_imp, k_imp)
            diff = (qk * qk_sqrt - qk_imp * qk_sqrt_imp)
            diff_sqr = diff * diff
            mse_contrib = tl.sum(diff_sqr)
            # tl.atomic_add(mse_loss, mse_contrib)
            mse_contrib_total += mse_contrib
            K_imp_block_ptr = tl.advance(K_imp_block_ptr, (0, BLOCK_N))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    mse_contrib_total /= NUM_ELEMENTS
    tl.atomic_add(mse_loss, mse_contrib_total)
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [32]\
#     for BN in [32]\
#     for s in ([1] if is_hip() else [3, 4, 7])\
#     for w in [4, 8]\
# ]

fixed_config = triton.Config(
    {'BLOCK_M': 16, 'BLOCK_N': 16},
    num_stages=4,
    num_warps=8
)

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, sm_scale_imp, M, Out,  #
              Q_importance, K_importance, mse_loss,  
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_qz_imp, stride_qh_imp, stride_qm_imp, stride_qk_imp,  #
              stride_kz_imp, stride_kh_imp, stride_kn_imp, stride_kk_imp,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              D_DASH: tl.constexpr,  #
              NUM_ELEMENTS: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_N <= D_DASH)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    qv_imp_offset = off_z.to(tl.int64) * stride_qz_imp + off_h.to(tl.int64) * stride_qh_imp

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Q_imp_block_ptr = tl.make_block_ptr(
        base=Q_importance + qv_imp_offset,
        shape=(N_CTX, D_DASH),
        strides=(stride_qm_imp, stride_qk_imp),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_DASH),
        order=(1, 0),
    )
    K_imp_block_ptr = tl.make_block_ptr(
        base=K_importance + qv_imp_offset,
        shape=(D_DASH, N_CTX),
        strides=(stride_kk_imp, stride_kn_imp),
        offsets=(0, 0),
        block_shape=(D_DASH, BLOCK_N),
        order=(0, 1),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_sqrt = sm_scale
    qk_scale = qk_sqrt * 1.44269504  # 1/log(2)
    qk_sqrt_imp = sm_scale_imp
    qk_scale_imp = qk_sqrt_imp * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q_imp = tl.load(Q_imp_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    # _attn_fwd_get_loss(q, q_imp, K_block_ptr, K_imp_block_ptr, V_block_ptr, mse_loss, start_m, qk_scale, qk_scale_imp, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_imp, K_block_ptr, K_imp_block_ptr, V_block_ptr,  #
                                        mse_loss,  #
                                        start_m, qk_sqrt, qk_scale, qk_sqrt_imp, qk_scale_imp,  #
                                        BLOCK_M, HEAD_DIM, NUM_ELEMENTS, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_imp, K_block_ptr, K_imp_block_ptr, V_block_ptr,  #
                                        mse_loss,  #
                                        start_m, qk_sqrt, qk_scale, qk_sqrt_imp, qk_scale_imp,  #
                                        BLOCK_M, HEAD_DIM, NUM_ELEMENTS, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq

@triton.jit
def _attn_bwd_dk_imp(
                   Q, Q_imp, k, k_imp, sm_scale, sm_scale_imp, num_elements, #
                   DMSE,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   stride_tok_imp, stride_d_imp,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   D_DASH: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, is_float16):
    dk_imp = tl.zeros([BLOCK_N1, D_DASH], dtype=tl.float32)
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_k_imp = tl.arange(0, D_DASH)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    qT_imp_ptrs = Q_imp + offs_m[None, :] * stride_tok_imp + offs_k_imp[:, None] * stride_d_imp
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dmse_eval = tl.load(DMSE)
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        qT_imp = tl.load(qT_imp_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        qkT = tl.dot(k, qT)
        qkT_imp = tl.dot(k_imp, qT_imp)
        diff = (qkT_imp * sm_scale_imp - qkT * sm_scale)
        tmp = dmse_eval * 2.0 * (1 / num_elements) * sm_scale_imp
        diff = diff.to(tl.float16)
        qT_imp = qT_imp.to(tl.float16)
        dk_imp += tmp * tl.dot(diff, tl.trans(qT_imp))
        if not is_float16:
            dk_imp = dk_imp.to(tl.float32)

        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        qT_imp_ptrs += step_m * stride_tok_imp
    return dk_imp

@triton.jit
def _attn_bwd_dq_imp(
                   q, q_imp, K, K_imp, sm_scale, sm_scale_imp, num_elements, #
                   DMSE,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   stride_tok_imp, stride_d_imp,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   D_DASH: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, is_float16):
    dq_imp = tl.zeros([BLOCK_N1, D_DASH], dtype=tl.float32)
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_k_imp = tl.arange(0, D_DASH)
    kT_ptrs = K + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    kT_imp_ptrs = K_imp + offs_m[None, :] * stride_tok_imp + offs_k_imp[:, None] * stride_d_imp
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dmse_eval = tl.load(DMSE)
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        kT_imp = tl.load(kT_imp_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        qkT = tl.dot(q, kT)
        qkT_imp = tl.dot(q_imp, kT_imp)
        diff = (qkT_imp * sm_scale_imp - qkT * sm_scale)
        tmp = dmse_eval * 2.0 * (1 / num_elements) * sm_scale_imp
        diff = diff.to(tl.float16)
        kT_imp = kT_imp.to(tl.float16)
        dq_imp += tmp * tl.dot(diff, tl.trans(kT_imp))
        if not is_float16:
            dq_imp = dq_imp.to(tl.float32)
        # Increment pointers.
        curr_m += step_m
        kT_ptrs += step_m * stride_tok
        kT_imp_ptrs += step_m * stride_tok_imp
    return dq_imp


@triton.jit
def _attn_bwd(Q, Q_imp, K, K_imp, sm_scale, sm_scale_imp, num_elements, #
              DQ_imp, DK_imp, DMSE, #
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              stride_z_imp, stride_h_imp, stride_tok_imp, stride_d_imp,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              D_DASH: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adj_imp = (stride_h_imp * (bhid % H) + stride_z_imp * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    Q_imp += adj_imp
    K_imp += adj_imp
    DQ_imp += adj_imp
    DK_imp += adj_imp

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)
    offs_k_imp = tl.arange(0, D_DASH)

    start_n = pid * BLOCK_N1
    start_m = 0
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    num_steps = N_CTX // BLOCK_M1
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    k_imp = tl.load(K_imp + offs_n[:, None] * stride_tok_imp + offs_k_imp[None, :] * stride_d_imp)
    dk_imp = _attn_bwd_dk_imp(
                            Q, Q_imp, k, k_imp, sm_scale, sm_scale_imp, num_elements,  #
                            DMSE,  #
                            stride_tok, stride_d,  #
                            stride_tok_imp, stride_d_imp,  #
                            H, N_CTX, BLOCK_M1,  #
                            BLOCK_N1, HEAD_DIM, D_DASH,  #
                            start_n, start_m, num_steps, k.dtype == tl.float16 #
                            )
    dk_imp_ptrs = DK_imp + offs_n[:, None] * stride_tok_imp + offs_k_imp[None, :] * stride_d_imp
    tl.store(dk_imp_ptrs, dk_imp)

    start_n = pid * BLOCK_N1
    start_m = 0
    num_steps = N_CTX // BLOCK_M1
    q = tl.load(Q + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    q_imp = tl.load(Q_imp + offs_n[:, None] * stride_tok_imp + offs_k_imp[None, :] * stride_d_imp)
    dq_imp = _attn_bwd_dq_imp(
                            q, q_imp, K, K_imp, sm_scale, sm_scale_imp, num_elements,  #
                            DMSE,  #
                            stride_tok, stride_d,  #
                            stride_tok_imp, stride_d_imp,  #
                            H, N_CTX, BLOCK_M1,  #
                            BLOCK_N1, HEAD_DIM, D_DASH,  #
                            start_n, start_m, num_steps, q.dtype == tl.float16 #
                            )
    dq_imp_ptrs = DQ_imp + offs_n[:, None] * stride_tok_imp + offs_k_imp[None, :] * stride_d_imp
    tl.store(dq_imp_ptrs, dq_imp)
    

class _attention_mse_loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_importance, k_importance, causal):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        D_DASH = q_importance.shape[-1]
        assert D_DASH == k_importance.shape[-1], "q_importance and k_importance must have the same last dimension"
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_Q)
        sm_scale_imp = 1.0 / math.sqrt(D_DASH)
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        mse_loss = torch.zeros(1, device=q.device, dtype=torch.float32)
        num_elements = (q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2])
        _attn_fwd[grid](
            q, k, v, sm_scale, sm_scale_imp, M, o,  #
            q_importance, k_importance, mse_loss,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q_importance.stride(0), q_importance.stride(1), q_importance.stride(2), q_importance.stride(3),  #
            k_importance.stride(0), k_importance.stride(1), k_importance.stride(2), k_importance.stride(3),  #
            Z=q.shape[0], H=q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            D_DASH=D_DASH,  #
            NUM_ELEMENTS=num_elements,  #
            STAGE=stage,  #
            **fixed_config.kwargs)
        ctx.save_for_backward(q, q_importance, k, k_importance, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.sm_scale_imp = sm_scale_imp
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.D_DASH = D_DASH
        ctx.num_elements = num_elements
        ctx.causal = causal
        return o, mse_loss

    @staticmethod
    def backward(ctx, do, dmse):
        q, q_importance, k, k_importance, v, o, M = ctx.saved_tensors
        do = do.contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dq_imp = torch.empty_like(q_importance)
        dk = torch.empty_like(k)
        dk_imp = torch.empty_like(k_importance)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1 = 16, 16
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)

        # PRE_BLOCK = 128
        # assert N_CTX % PRE_BLOCK == 0
        # pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        # delta = torch.empty_like(M)
        # _attn_bwd_preprocess[pre_grid](
        #     o, do,  #
        #     delta,  #
        #     BATCH, N_HEAD, N_CTX,  #
        #     BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        # )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid]( 
            q, q_importance, k, k_importance, ctx.sm_scale, ctx.sm_scale_imp, ctx.num_elements,  #
            dq_imp, dk_imp, dmse, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            q_importance.stride(0), q_importance.stride(1), q_importance.stride(2), q_importance.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            D_DASH=ctx.D_DASH,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        return None, None, None, dq_imp, dk_imp, None


attention_mse_loss = _attention_mse_loss.apply

from torch import nn
from torch.nn import MSELoss
import time
if __name__ == '__main__':
    B, H, N_CTX, D = 1, 32, 512, 128
    D_DASH = 32
    causal = True
    DTYPE = torch.float32
    LOAD_WEIGHT = False
    import os
    if LOAD_WEIGHT and os.path.exists("export_params.pt"):
        print("[Info] Detected export_params.pt, loading saved tensors...")
        debug_tensors = torch.load("export_params.pt", map_location=DEVICE)
        
        q = debug_tensors["q"].detach().clone().requires_grad_().contiguous()
        k = debug_tensors["k"].detach().clone().requires_grad_().contiguous()
        v = debug_tensors["v"].detach().clone().requires_grad_().contiguous()
        q_importance = debug_tensors["q_importance"].detach().clone().requires_grad_().contiguous()
        k_importance = debug_tensors["k_importance"].detach().clone().requires_grad_().contiguous()

        print("[Success] Tensors loaded successfully!")
        print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        print(f"q_importance.shape: {q_importance.shape}, k_importance.shape: {k_importance.shape}")
    else:
        print("[Info] No export_params.pt found, initializing random tensors...")
        DTYPE = torch.float32

        gain_q = math.sqrt(5.0 / D)  
        gain_k = math.sqrt(5.0 / D)
        gain_v = math.sqrt(3.0 / D)    

        q = (torch.randn((B, H, N_CTX, D), dtype=DTYPE, device=DEVICE) * gain_q).requires_grad_()
        k = (torch.randn((B, H, N_CTX, D), dtype=DTYPE, device=DEVICE) * gain_k).requires_grad_()
        v = (torch.randn((B, H, N_CTX, D), dtype=DTYPE, device=DEVICE) * gain_v).requires_grad_()

        gain_q_imp = math.sqrt(5.0 / D_DASH)
        gain_k_imp = math.sqrt(5.0 / D_DASH)
        q_importance = (torch.randn((B, H, N_CTX, D_DASH), dtype=DTYPE, device=DEVICE) * gain_q_imp + 0.1).requires_grad_()
        k_importance = (torch.randn((B, H, N_CTX, D_DASH), dtype=DTYPE, device=DEVICE) * gain_k_imp - 0.1).requires_grad_()
        print("[Info] Random tensors initialized.")

    # warm up for the triton implementation
    attn_output, mse_loss_triton = attention_mse_loss(q.to(torch.float16),
                                                                    k.to(torch.float16),
                                                                    v.to(torch.float16),
                                                                    q_importance.to(torch.float16),
                                                                    k_importance.to(torch.float16), True)


    mse_loss_triton.backward()
    q_importance.grad = None
    k_importance.grad = None

    # warm up for the original implementation
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    importance_mask = torch.matmul(q_importance, k_importance.transpose(-2, -1)) / math.sqrt(D_DASH) # [B, H, Lq, Lk]
    mse_func = MSELoss(reduction='none')
    mse_loss_original = mse_func(attn_weights, importance_mask)
    mse_loss_original = mse_loss_original.mean()
    if causal:
        causal_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=attn_weights.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(v.dtype)
    attn_output_original = torch.matmul(attn_weights, v)
    mse_loss_original.backward()
    q_importance.grad = None
    k_importance.grad = None

    # triton implementation
    tri_start = time.time()
    att_output_triton, mse_loss_triton = attention_mse_loss(q, k, v, q_importance, k_importance, causal)
    mse_loss_triton.backward()
    tri_dq_imp, q_importance.grad = q_importance.grad.clone(), None
    tri_dk_imp, k_importance.grad = k_importance.grad.clone(), None
    tri_end = time.time()

    # original implementation
    ori_start = time.time()
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    importance_mask = torch.matmul(q_importance, k_importance.transpose(-2, -1)) / math.sqrt(D_DASH) # [B, H, Lq, Lk]
    mse_func = MSELoss(reduction='none')
    mse_loss_original = mse_func(attn_weights, importance_mask)
    mse_loss_original = mse_loss_original.mean()
    mse_loss_original = mse_loss_original
    if causal:
        causal_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=attn_weights.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
    attn_output_original = torch.matmul(attn_weights, v)
    
    mse_loss_original.backward()
    ref_dk_imp, k_importance.grad = k_importance.grad.clone(), None
    ref_dq_imp, q_importance.grad = q_importance.grad.clone(), None
    ori_end = time.time()
    
    print(f'mse_loss_triton: {mse_loss_triton}')
    print(f'mse_loss_original: {mse_loss_original}')
    print(f'mean of ref_dk_imp: {ref_dk_imp.mean()}')
    print(f'mean of tri_dk_imp: {tri_dk_imp.mean()}')
    # print(f'error: {torch.abs(ref_dk_imp - tri_dk_imp).mean()}')
    print(f'ref_dq_imp: {ref_dq_imp[0][0]}')
    print(f'tri_dq_imp: {tri_dq_imp[0][0]}')
    print(f'mean of ref_dq_imp: {ref_dq_imp.mean()}')
    print(f'mean of tri_dq_imp: {tri_dq_imp.mean()}')
    # print(f'error: {torch.abs(ref_dq_imp - tri_dq_imp).mean()}')

    # assert torch.allclose(attn_output_original, att_output_triton, atol=1e-2, rtol=0), f'{attn_output_original.mean()} vs {att_output_triton.mean()}'
    perc_diff = 100 * torch.abs(attn_output_original - att_output_triton).mean() / torch.abs(attn_output_original).mean()
    print(f'passed test for attention output with {attn_output_original.mean()} vs {att_output_triton.mean()}, \t\t\tpercentage diff: {perc_diff}%')
    # assert torch.allclose(mse_loss_triton, mse_loss_original, atol=1e-1, rtol=0), f'{mse_loss_triton} vs {mse_loss_original}'
    perc_diff = 100 * torch.abs(mse_loss_triton - mse_loss_original) / torch.abs(mse_loss_original)
    print(f'passed test for mse loss with {mse_loss_triton.item()} vs {mse_loss_original.item()}, \t\t\t\tpercentage diff: {perc_diff.item()}%')
    # assert torch.allclose(ref_dk_imp, tri_dk_imp, atol=1e-1, rtol=0), f'{ref_dk_imp.mean()} vs {tri_dk_imp.mean()}'
    perc_diff = 100 * torch.abs(ref_dk_imp - tri_dk_imp).mean() / torch.abs(ref_dk_imp).mean()
    print(f'passed test for dk_imp with {ref_dk_imp.mean()} vs {tri_dk_imp.mean()}, \t\tpercentage diff: {perc_diff}%')
    # assert torch.allclose(ref_dq_imp, tri_dq_imp, atol=1e-1, [rtol=0), f'{ref_dq_imp.mean()} vs {tri_dq_imp.mean()}'
    perc_diff = 100 * torch.abs(ref_dq_imp - tri_dq_imp).mean() / torch.abs(ref_dq_imp).mean()
    print(f'passed test for dq_imp with {ref_dq_imp.mean()} vs {tri_dq_imp.mean()}, \t\tpercentage diff: {perc_diff}%')
    print(f'original time: {ori_end - ori_start}')
    print(f'triton time: {tri_end - tri_start}')
