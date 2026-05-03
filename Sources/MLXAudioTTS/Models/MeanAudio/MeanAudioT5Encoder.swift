//
//  MeanAudioT5Encoder.swift
//  MLXAudio
//
//  T5 text encoder for MeanAudio: flan-t5-large → (1, 77, 1024) features.
//  Adapted from SAMAudioTextEncoder.swift (same T5 architecture, different usage).
//

import Foundation
import Hub
@preconcurrency import MLX
import MLXAudioCore
import MLXNN
import MLXLMCommon
import Tokenizers

// MARK: - T5 Config

private struct MAT5Config: Codable {
    var vocabSize: Int
    var dModel: Int
    var dKV: Int
    var dFF: Int
    var numLayers: Int
    var numHeads: Int
    var relativeAttentionNumBuckets: Int
    var relativeAttentionMaxDistance: Int
    var dropoutRate: Float
    var layerNormEpsilon: Float
    var isGatedAct: Bool
    var denseActFn: String

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case dKV = "d_kv"
        case dFF = "d_ff"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case dropoutRate = "dropout_rate"
        case layerNormEpsilon = "layer_norm_epsilon"
        case isGatedAct = "is_gated_act"
        case denseActFn = "dense_act_fn"
    }

    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32128
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1024
        dKV = try c.decodeIfPresent(Int.self, forKey: .dKV) ?? 64
        dFF = try c.decodeIfPresent(Int.self, forKey: .dFF) ?? 2816
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 24
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        relativeAttentionNumBuckets = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionNumBuckets) ?? 32
        relativeAttentionMaxDistance = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionMaxDistance) ?? 128
        dropoutRate = try c.decodeIfPresent(Float.self, forKey: .dropoutRate) ?? 0.1
        layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-6
        isGatedAct = try c.decodeIfPresent(Bool.self, forKey: .isGatedAct) ?? true
        denseActFn = try c.decodeIfPresent(String.self, forKey: .denseActFn) ?? "gelu_new"
    }
}

// MARK: - T5 Layer Norm (RMS-style)

private final class MAT5LayerNorm: Module {
    let eps: Float
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(hiddenSize: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let variance = mean(hiddenStates.asType(.float32) * hiddenStates.asType(.float32), axis: -1, keepDims: true)
        let normed = hiddenStates * rsqrt(variance + MLXArray(eps))
        return weight * normed
    }
}

// MARK: - T5 Feed-Forward

private final class MAT5DenseActDense: Module {
    let actFn: String
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: MAT5Config) {
        self.actFn = config.denseActFn
        self._wi.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wo.wrappedValue = Linear(config.dFF, config.dModel, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = wi(x)
        h = actFn == "relu" ? relu(h) : gelu(h)
        return wo(h)
    }
}

private final class MAT5DenseGatedActDense: Module {
    let actFn: String
    @ModuleInfo(key: "wi_0") var wi0: Linear
    @ModuleInfo(key: "wi_1") var wi1: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(config: MAT5Config) {
        self.actFn = config.denseActFn
        self._wi0.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wi1.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wo.wrappedValue = Linear(config.dFF, config.dModel, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gated = wi0(x)
        gated = actFn == "relu" ? relu(gated) : gelu(gated)
        return wo(gated * wi1(x))
    }
}

private final class MAT5LayerFF: Module {
    @ModuleInfo(key: "DenseReluDense") var denseReluDense: Module
    @ModuleInfo(key: "layer_norm") var layerNorm: MAT5LayerNorm

    init(config: MAT5Config) {
        if config.isGatedAct {
            self._denseReluDense.wrappedValue = MAT5DenseGatedActDense(config: config)
        } else {
            self._denseReluDense.wrappedValue = MAT5DenseActDense(config: config)
        }
        self._layerNorm.wrappedValue = MAT5LayerNorm(hiddenSize: config.dModel, eps: config.layerNormEpsilon)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let normed = layerNorm(hiddenStates)
        let out: MLXArray
        if let gated = denseReluDense as? MAT5DenseGatedActDense {
            out = gated(normed)
        } else if let dense = denseReluDense as? MAT5DenseActDense {
            out = dense(normed)
        } else {
            out = normed
        }
        return hiddenStates + out
    }
}

// MARK: - T5 Attention

private final class MAT5Attention: Module {
    let hasRelativeAttentionBias: Bool
    let relativeAttentionNumBuckets: Int
    let relativeAttentionMaxDistance: Int
    let dModel: Int
    let keyValueProjDim: Int
    let nHeads: Int
    let innerDim: Int

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?

    init(config: MAT5Config, hasRelativeAttentionBias: Bool = false) {
        self.hasRelativeAttentionBias = hasRelativeAttentionBias
        self.relativeAttentionNumBuckets = config.relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = config.relativeAttentionMaxDistance
        self.dModel = config.dModel
        self.keyValueProjDim = config.dKV
        self.nHeads = config.numHeads
        self.innerDim = nHeads * keyValueProjDim

        self._q.wrappedValue = Linear(dModel, innerDim, bias: false)
        self._k.wrappedValue = Linear(dModel, innerDim, bias: false)
        self._v.wrappedValue = Linear(dModel, innerDim, bias: false)
        self._o.wrappedValue = Linear(innerDim, dModel, bias: false)
        self._relativeAttentionBias.wrappedValue = hasRelativeAttentionBias
            ? Embedding(embeddingCount: relativeAttentionNumBuckets, dimensions: nHeads)
            : nil
    }

    private func relativePositionBucket(
        relativePosition: Int, bidirectional: Bool = true,
        numBuckets: Int, maxDistance: Int
    ) -> Int {
        var rp = relativePosition
        var buckets = numBuckets
        var relativeBuckets = 0

        if bidirectional {
            buckets /= 2
            if rp > 0 { relativeBuckets += buckets }
            rp = abs(rp)
        } else {
            rp = max(-rp, 0)
        }

        let maxExact = buckets / 2
        if rp < maxExact { return relativeBuckets + rp }

        let rpF = Float(rp), meF = Float(maxExact), mdF = Float(maxDistance)
        let b = meF + Foundation.log(rpF / meF) / Foundation.log(mdF / meF) * Float(buckets - maxExact)
        return relativeBuckets + min(Int(b), buckets - 1)
    }

    private func computeBias(queryLength: Int, keyLength: Int) -> MLXArray {
        guard let relativeAttentionBias else {
            return MLXArray.zeros([1, nHeads, queryLength, keyLength])
        }
        var buckets: [Int] = []
        buckets.reserveCapacity(queryLength * keyLength)
        for q in 0..<queryLength {
            for k in 0..<keyLength {
                buckets.append(relativePositionBucket(
                    relativePosition: k - q,
                    numBuckets: relativeAttentionNumBuckets,
                    maxDistance: relativeAttentionMaxDistance
                ))
            }
        }
        let arr = MLXArray(buckets, [queryLength, keyLength]).asType(.int32)
        return relativeAttentionBias(arr).transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, mask: MLXArray? = nil, positionBias: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let B = hiddenStates.shape[0], S = hiddenStates.shape[1]

        var qs = q(hiddenStates).reshaped([B, S, nHeads, keyValueProjDim]).transposed(0, 2, 1, 3)
        var ks = k(hiddenStates).reshaped([B, S, nHeads, keyValueProjDim]).transposed(0, 2, 1, 3)
        var vs = v(hiddenStates).reshaped([B, S, nHeads, keyValueProjDim]).transposed(0, 2, 1, 3)

        var scores = matmul(qs, ks.transposed(0, 1, 3, 2))

        let bias: MLXArray
        if let positionBias { bias = positionBias }
        else if hasRelativeAttentionBias { bias = computeBias(queryLength: S, keyLength: S) }
        else { bias = MLXArray.zeros([1, nHeads, S, S]) }

        scores = scores + bias
        if let mask { scores = scores + mask }

        let weights = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        var out = matmul(weights, vs)
        out = out.transposed(0, 2, 1, 3).reshaped([B, S, innerDim])
        return (o(out), bias)
    }
}

// MARK: - T5 Encoder Blocks

private final class MAT5LayerSelfAttention: Module {
    @ModuleInfo(key: "SelfAttention") var selfAttention: MAT5Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: MAT5LayerNorm

    init(config: MAT5Config, hasRelativeAttentionBias: Bool = false) {
        self._selfAttention.wrappedValue = MAT5Attention(config: config, hasRelativeAttentionBias: hasRelativeAttentionBias)
        self._layerNorm.wrappedValue = MAT5LayerNorm(hiddenSize: config.dModel, eps: config.layerNormEpsilon)
    }

    func callAsFunction(_ h: MLXArray, mask: MLXArray? = nil, bias: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (attn, b) = selfAttention(layerNorm(h), mask: mask, positionBias: bias)
        return (h + attn, b)
    }
}

private final class MAT5Block: Module {
    @ModuleInfo(key: "self_attention") var selfAttention: MAT5LayerSelfAttention
    @ModuleInfo(key: "ff") var ff: MAT5LayerFF

    init(config: MAT5Config, hasRelativeAttentionBias: Bool = false) {
        self._selfAttention.wrappedValue = MAT5LayerSelfAttention(config: config, hasRelativeAttentionBias: hasRelativeAttentionBias)
        self._ff.wrappedValue = MAT5LayerFF(config: config)
    }

    func callAsFunction(_ h: MLXArray, mask: MLXArray? = nil, bias: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (attended, b) = selfAttention(h, mask: mask, bias: bias)
        return (ff(attended), b)
    }
}

private final class MAT5Stack: Module {
    private var embedTokens: Embedding?
    @ModuleInfo(key: "block") var block: [MAT5Block]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: MAT5LayerNorm

    init(config: MAT5Config) {
        self._block.wrappedValue = (0..<config.numLayers).map { i in
            MAT5Block(config: config, hasRelativeAttentionBias: i == 0)
        }
        self._finalLayerNorm.wrappedValue = MAT5LayerNorm(hiddenSize: config.dModel, eps: config.layerNormEpsilon)
    }

    func setInputEmbeddings(_ e: Embedding) { embedTokens = e }

    func callAsFunction(inputIDs: MLXArray? = nil, attentionMask: MLXArray? = nil, inputsEmbeds: MLXArray? = nil) -> MLXArray {
        let embeds: MLXArray
        if let inputsEmbeds { embeds = inputsEmbeds }
        else if let inputIDs, let embedTokens { embeds = embedTokens(inputIDs) }
        else { fatalError("Must provide inputIDs or inputsEmbeds") }

        var mask: MLXArray? = nil
        if let attentionMask {
            var m = attentionMask.asType(.float32).expandedDimensions(axis: 1).expandedDimensions(axis: 1)
            m = (MLXArray(1.0) - m) * MLXArray(-1e9)
            mask = m
        }

        var h = embeds
        var bias: MLXArray? = nil
        for layer in block {
            (h, bias) = layer(h, mask: mask, bias: bias)
        }
        return finalLayerNorm(h)
    }
}

private final class MAT5Encoder: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: MAT5Stack

    init(config: MAT5Config) {
        let emb = Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel)
        let stack = MAT5Stack(config: config)
        stack.setInputEmbeddings(emb)
        self._shared.wrappedValue = emb
        self._encoder.wrappedValue = stack
        super.init()
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        encoder(inputIDs: inputIDs, attentionMask: attentionMask)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("decoder.") || key.hasPrefix("lm_head.") { continue }
            var mapped = key
            if key == "encoder.embed_tokens.weight" { mapped = "shared.weight" }
            mapped = mapped.replacingOccurrences(of: ".layer.0.", with: ".self_attention.")
            mapped = mapped.replacingOccurrences(of: ".layer.1.", with: ".ff.")
            sanitized[mapped] = value
        }
        return sanitized
    }

    static func fromPretrained(modelFolder: URL) throws -> MAT5Encoder {
        let configData = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MAT5Config.self, from: configData)
        let model = MAT5Encoder(config: config)

        let files = try FileManager.default.contentsOfDirectory(at: modelFolder, includingPropertiesForKeys: nil)
        var weights: [String: MLXArray] = [:]
        for file in files.filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            weights.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }

        let sanitized = sanitize(weights: weights)
        let params = ModuleParameters.unflattened(sanitized)
        model.update(parameters: params)
        eval(model)
        return model
    }
}

// MARK: - High-level T5 Text Encoder for MeanAudio

public final class MeanAudioT5Encoder: @unchecked Sendable {
    private var model: MAT5Encoder?
    private var tokenizer: Tokenizers.Tokenizer?
    private let modelFolder: URL
    private let maxLength: Int

    public init(modelFolder: URL, maxLength: Int = 77) {
        self.modelFolder = modelFolder
        self.maxLength = maxLength
    }

    public func ensureLoaded() async throws {
        guard model == nil else { return }
        self.model = try MAT5Encoder.fromPretrained(modelFolder: modelFolder)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    public func encode(_ texts: [String]) async throws -> MLXArray {
        try await ensureLoaded()
        guard let model, let tokenizer else {
            throw T5EncoderError.notLoaded
        }

        let tokenized = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
        let batch = tokenized.count

        var idsFlat: [Int32] = []
        var maskFlat: [Int32] = []
        idsFlat.reserveCapacity(batch * maxLength)
        maskFlat.reserveCapacity(batch * maxLength)

        for var ids in tokenized {
            if ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
            let valid = ids.count
            let padded = ids + Array(repeating: 0, count: max(0, maxLength - valid))
            idsFlat.append(contentsOf: padded.prefix(maxLength).map { Int32($0) })
            maskFlat.append(contentsOf: (0..<maxLength).map { Int32($0 < valid ? 1 : 0) })
        }

        let inputIDs = MLXArray(idsFlat, [batch, maxLength])
        let attentionMask = MLXArray(maskFlat, [batch, maxLength]).asType(.bool)

        let features = model(inputIDs: inputIDs, attentionMask: attentionMask)
        eval(features)
        return features
    }

    public var isLoaded: Bool { model != nil }
}

enum T5EncoderError: Error, LocalizedError {
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .notLoaded: return "T5 text encoder not loaded"
        }
    }
}
