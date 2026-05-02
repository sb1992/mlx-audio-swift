import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - RoPE (Llama3 scaled)

enum TadaRoPE {
    static func computeLlama3InvFreq(
        headDim: Int,
        theta: Float,
        factor: Float,
        lowFreqFactor: Float,
        highFreqFactor: Float,
        originalMaxPositionEmbeddings: Int
    ) -> MLXArray {
        let invFreq = 1.0 / pow(
            MLXArray(theta),
            MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) }) / Float(headDim)
        )
        let oldCtx = Float(originalMaxPositionEmbeddings)
        let lowWavelen = oldCtx / lowFreqFactor
        let highWavelen = oldCtx / highFreqFactor
        var newFreqs = [Float]()

        for i in 0..<invFreq.shape[0] {
            let freq: Float = invFreq[i].item()
            let wavelen = 2.0 * Float.pi / freq
            if wavelen < highWavelen {
                newFreqs.append(freq)
            } else if wavelen > lowWavelen {
                newFreqs.append(freq / factor)
            } else {
                let smooth = (oldCtx / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
                newFreqs.append((1 - smooth) * freq / factor + smooth * freq)
            }
        }
        return MLXArray(newFreqs)
    }

    static func buildRopeCache(
        seqLen: Int,
        headDim: Int,
        ropeTheta: Float,
        ropeScaling: TadaRopeScaling
    ) -> (cos: MLXArray, sin: MLXArray) {
        let invFreq = computeLlama3InvFreq(
            headDim: headDim,
            theta: ropeTheta,
            factor: ropeScaling.factor,
            lowFreqFactor: ropeScaling.lowFreqFactor,
            highFreqFactor: ropeScaling.highFreqFactor,
            originalMaxPositionEmbeddings: ropeScaling.originalMaxPositionEmbeddings
        )
        let positions = MLXArray(0..<seqLen).asType(.float32)
        let freqs = outer(positions, invFreq)
        return (cos(freqs), sin(freqs))
    }

    static func applyRope(_ x: MLXArray, cos cosArr: MLXArray, sin sinArr: MLXArray) -> MLXArray {
        let hdim = x.shape[x.ndim - 1]
        let x0 = x[.ellipsis, ..<(hdim / 2)]
        let x1 = x[.ellipsis, (hdim / 2)...]
        let cosE = expandedDimensions(expandedDimensions(cosArr, axis: 0), axis: 0)
        let sinE = expandedDimensions(expandedDimensions(sinArr, axis: 0), axis: 0)
        return concatenated([x0 * cosE - x1 * sinE, x0 * sinE + x1 * cosE], axis: -1)
    }
}

// MARK: - KV Cache

class TadaKVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int = 0

    func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        if let existing = keys {
            keys = concatenated([existing, k], axis: 2)
            values = concatenated([values!, v], axis: 2)
        } else {
            keys = k
            values = v
        }
        offset = keys!.shape[2]
        return (keys!, values!)
    }

    var seqLen: Int { offset }
}

// MARK: - Attention (GQA)

class TadaAttention: Module {
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(config: TadaConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: TadaKVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])
        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKvHeads, headDim).transposed(0, 2, 1, 3)

        q = TadaRoPE.applyRope(q, cos: cos, sin: sin)
        k = TadaRoPE.applyRope(k, cos: cos, sin: sin)

        var kFinal = k
        var vFinal = v
        if let cache {
            (kFinal, vFinal) = cache.update(keys: k, values: v)
        }

        let nRep = numHeads / numKvHeads
        if nRep > 1 {
            kFinal = repeated(kFinal, count: nRep, axis: 1)
            vFinal = repeated(vFinal, count: nRep, axis: 1)
        }

        var attn = q.matmul(kFinal.transposed(0, 1, 3, 2)) * scale
        if let mask { attn = attn + mask }
        attn = softmax(attn, axis: -1)
        let out = attn.matmul(vFinal).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(out)
    }
}

// MARK: - MLP

class TadaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: TadaConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Transformer Block

class TadaTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TadaAttention
    var mlp: TadaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(config: TadaConfig) {
        self._selfAttn.wrappedValue = TadaAttention(config: config)
        self.mlp = TadaMLP(config: config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: TadaKVCache? = nil
    ) -> MLXArray {
        let r1 = selfAttn(inputLayernorm(x), cos: cos, sin: sin, mask: mask, cache: cache)
        let h = x + r1
        let r2 = mlp(postAttentionLayernorm(h))
        return h + r2
    }
}

// MARK: - Llama Model

class TadaLlamaModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [TadaTransformerBlock]
    var norm: RMSNorm

    init(config: TadaConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map { _ in TadaTransformerBlock(config: config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXArray? = nil,
        cache: [TadaKVCache]? = nil
    ) -> MLXArray {
        var h = inputsEmbeds
        for (i, layer) in layers.enumerated() {
            h = layer(h, cos: cos, sin: sin, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}
