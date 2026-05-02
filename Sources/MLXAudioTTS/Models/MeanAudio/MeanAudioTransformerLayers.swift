//
//  MeanAudioTransformerLayers.swift
//  MLXAudio
//
//  Transformer building blocks for MeanAudio: SelfAttention, MMDiT blocks.
//  Ported from MeanAudio Python: meanaudio/model/transformer_layers.py
//

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - AdaLN Modulation

func maModulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

// MARK: - Self Attention

class MASelfAttention: Module {
    let dim: Int
    let nheads: Int

    @ModuleInfo var qkv: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MARMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MARMSNorm

    init(dim: Int, nheads: Int) {
        self.dim = dim
        self.nheads = nheads

        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._qNorm.wrappedValue = MARMSNorm(dim: dim / nheads)
        self._kNorm.wrappedValue = MARMSNorm(dim: dim / nheads)
        super.init()
    }

    func preAttention(_ x: MLXArray, rot: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        let B = x.dim(0)
        let N = x.dim(1)
        let headDim = dim / nheads

        // qkv: (B, N, 3*D) → (B, N, nheads, headDim, 3)
        let qkvOut = qkv(x)
            .reshaped(B, N, nheads, headDim, 3)

        // Split into q, k, v: each (B, N, nheads, headDim)
        var q = qkvOut[.ellipsis, 0]
        var k = qkvOut[.ellipsis, 1]
        let v = qkvOut[.ellipsis, 2]

        // RMSNorm on q, k
        q = qNorm(q)
        k = kNorm(k)

        // Transpose to (B, nheads, N, headDim) for attention
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        // Apply RoPE if provided
        if let rot = rot {
            q = MARoPE.applyRoPE(q, rot: rot)
            k = MARoPE.applyRoPE(k, rot: rot)
        }

        return (q, k, vT)
    }
}

// MARK: - RMS Norm (no learnable weight, elementwise_affine=False equivalent)

class MARMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dim: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Layer Norm (elementwise_affine=False)

class MALayerNorm: Module {
    let dims: Int
    let eps: Float

    init(dims: Int, eps: Float = 1e-5) {
        self.dims = dims
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) / (variance + MLXArray(eps)).sqrt()
    }
}

// MARK: - Attention helper

func maAttention(q: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
    // q, k, v: (B, nheads, N, headDim)
    let scale = sqrt(Float(q.dim(3)))
    let out = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v, scale: 1.0 / scale, mask: .none
    )
    // out: (B, nheads, N, headDim) → (B, N, nheads * headDim)
    let B = out.dim(0)
    let N = out.dim(2)
    return out.transposed(0, 2, 1, 3).reshaped(B, N, -1)
}

// MARK: - MMDiT Single Block

class MAMMDitSingleBlock: Module {
    let preOnly: Bool
    let useConv: Bool

    @ModuleInfo var norm1: MALayerNorm
    @ModuleInfo var attn: MASelfAttention
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: MAAdaLNModulation

    @ModuleInfo var linear1: Module?
    @ModuleInfo var norm2: MALayerNorm?
    @ModuleInfo var ffn: Module?

    init(dim: Int, nhead: Int, mlpRatio: Float = 4.0, preOnly: Bool = false, kernelSize: Int = 7, padding: Int = 3) {
        self.preOnly = preOnly
        self.useConv = kernelSize != 1

        self._norm1.wrappedValue = MALayerNorm(dims: dim)
        self._attn.wrappedValue = MASelfAttention(dim: dim, nheads: nhead)

        if preOnly {
            self._adaLNModulation.wrappedValue = MAAdaLNModulation(dim: dim, numCoeffs: 2)
            self._linear1.wrappedValue = nil
            self._norm2.wrappedValue = nil
            self._ffn.wrappedValue = nil
        } else {
            self._adaLNModulation.wrappedValue = MAAdaLNModulation(dim: dim, numCoeffs: 6)

            if kernelSize == 1 {
                self._linear1.wrappedValue = Linear(dim, dim)
                self._ffn.wrappedValue = MAMLP(dim: dim, hiddenDim: Int(Float(dim) * mlpRatio))
            } else {
                self._linear1.wrappedValue = MAChannelLastConv1d(
                    inChannels: dim, outChannels: dim, kernelSize: kernelSize, padding: padding
                )
                self._ffn.wrappedValue = MAConvMLP(
                    dim: dim, hiddenDim: Int(Float(dim) * mlpRatio),
                    kernelSize: kernelSize, padding: padding
                )
            }
            self._norm2.wrappedValue = MALayerNorm(dims: dim)
        }
        super.init()
    }

    struct PreAttentionResult {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let gateMSA: MLXArray?
        let shiftMLP: MLXArray?
        let scaleMLP: MLXArray?
        let gateMLP: MLXArray?
    }

    func preAttention(_ x: MLXArray, c: MLXArray, rot: MLXArray?) -> PreAttentionResult {
        let modulation = adaLNModulation(c)

        let shiftMSA: MLXArray
        let scaleMSA: MLXArray
        let gateMSA: MLXArray?
        let shiftMLP: MLXArray?
        let scaleMLP: MLXArray?
        let gateMLP: MLXArray?

        if preOnly {
            let chunks = modulation.split(parts: 2, axis: -1)
            shiftMSA = chunks[0]
            scaleMSA = chunks[1]
            gateMSA = nil; shiftMLP = nil; scaleMLP = nil; gateMLP = nil
        } else {
            let chunks = modulation.split(parts: 6, axis: -1)
            shiftMSA = chunks[0]; scaleMSA = chunks[1]; gateMSA = chunks[2]
            shiftMLP = chunks[3]; scaleMLP = chunks[4]; gateMLP = chunks[5]
        }

        let xMod = maModulate(norm1(x), shift: shiftMSA, scale: scaleMSA)
        let (q, k, v) = attn.preAttention(xMod, rot: rot)

        return PreAttentionResult(
            q: q, k: k, v: v,
            gateMSA: gateMSA, shiftMLP: shiftMLP, scaleMLP: scaleMLP, gateMLP: gateMLP
        )
    }

    private func applyLinear1(_ attnOut: MLXArray) -> MLXArray {
        if useConv, let conv = linear1 as? MAChannelLastConv1d {
            return conv(attnOut)
        } else if let lin = linear1 as? Linear {
            return lin(attnOut)
        }
        return attnOut
    }

    private func applyFfn(_ r: MLXArray) -> MLXArray {
        if useConv, let convMlp = ffn as? MAConvMLP {
            return convMlp(r)
        } else if let mlp = ffn as? MAMLP {
            return mlp(r)
        }
        return r
    }

    func postAttention(_ x: MLXArray, attnOut: MLXArray, gateMSA: MLXArray?, shiftMLP: MLXArray?, scaleMLP: MLXArray?, gateMLP: MLXArray?) -> MLXArray {
        if preOnly { return x }

        guard let gateMSA, let shiftMLP, let scaleMLP, let gateMLP, let norm2 else { return x }

        var out = x + applyLinear1(attnOut) * gateMSA
        let r = maModulate(norm2(out), shift: shiftMLP, scale: scaleMLP)
        out = out + applyFfn(r) * gateMLP

        return out
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray, rot: MLXArray?) -> MLXArray {
        let pre = preAttention(x, c: cond, rot: rot)
        let attnOut = maAttention(q: pre.q, k: pre.k, v: pre.v)
        return postAttention(x, attnOut: attnOut, gateMSA: pre.gateMSA, shiftMLP: pre.shiftMLP, scaleMLP: pre.scaleMLP, gateMLP: pre.gateMLP)
    }
}

// MARK: - AdaLN Modulation (SiLU → Linear)

class MAAdaLNModulation: Module {
    @ModuleInfo var linear: Linear

    init(dim: Int, numCoeffs: Int) {
        self._linear.wrappedValue = Linear(dim, numCoeffs * dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(silu(x))
    }
}

// MARK: - Joint Block

class MAJointBlock: Module {
    let preOnly: Bool
    @ModuleInfo(key: "latent_block") var latentBlock: MAMMDitSingleBlock
    @ModuleInfo(key: "text_block") var textBlock: MAMMDitSingleBlock

    init(dim: Int, nhead: Int, mlpRatio: Float = 4.0, preOnly: Bool = false) {
        self.preOnly = preOnly
        self._latentBlock.wrappedValue = MAMMDitSingleBlock(
            dim: dim, nhead: nhead, mlpRatio: mlpRatio, preOnly: false,
            kernelSize: 3, padding: 1
        )
        self._textBlock.wrappedValue = MAMMDitSingleBlock(
            dim: dim, nhead: nhead, mlpRatio: mlpRatio, preOnly: preOnly,
            kernelSize: 1, padding: 0
        )
        super.init()
    }

    func callAsFunction(
        latent: MLXArray, textF: MLXArray,
        globalC: MLXArray, extendedC: MLXArray,
        latentRot: MLXArray?, textRot: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let xPre = latentBlock.preAttention(latent, c: extendedC, rot: latentRot)
        let tPre = textBlock.preAttention(textF, c: globalC, rot: textRot)

        // Joint attention: concatenate Q, K, V across sequence dim
        let jointQ = MLX.concatenated([xPre.q, tPre.q], axis: 2)
        let jointK = MLX.concatenated([xPre.k, tPre.k], axis: 2)
        let jointV = MLX.concatenated([xPre.v, tPre.v], axis: 2)

        let attnOut = maAttention(q: jointQ, k: jointK, v: jointV)

        let latentLen = latent.dim(1)
        let xAttnOut = attnOut[0..., ..<latentLen]
        let tAttnOut = attnOut[0..., latentLen...]

        let newLatent = latentBlock.postAttention(
            latent, attnOut: xAttnOut,
            gateMSA: xPre.gateMSA, shiftMLP: xPre.shiftMLP,
            scaleMLP: xPre.scaleMLP, gateMLP: xPre.gateMLP
        )

        var newTextF = textF
        if !preOnly {
            newTextF = textBlock.postAttention(
                textF, attnOut: tAttnOut,
                gateMSA: tPre.gateMSA, shiftMLP: tPre.shiftMLP,
                scaleMLP: tPre.scaleMLP, gateMLP: tPre.gateMLP
            )
        }

        return (newLatent, newTextF)
    }
}

// MARK: - Final Block

class MAFinalBlock: Module {
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: MAAdaLNModulation
    @ModuleInfo var norm: MALayerNorm
    @ModuleInfo var conv: MAChannelLastConv1d

    init(dim: Int, outDim: Int) {
        self._adaLNModulation.wrappedValue = MAAdaLNModulation(dim: dim, numCoeffs: 2)
        self._norm.wrappedValue = MALayerNorm(dims: dim)
        self._conv.wrappedValue = MAChannelLastConv1d(
            inChannels: dim, outChannels: outDim, kernelSize: 7, padding: 3
        )
        super.init()
    }

    func callAsFunction(_ latent: MLXArray, c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation(c)
        let chunks = modulation.split(parts: 2, axis: -1)
        let shift = chunks[0]
        let scale = chunks[1]
        let out = maModulate(norm(latent), shift: shift, scale: scale)
        return conv(out)
    }
}
