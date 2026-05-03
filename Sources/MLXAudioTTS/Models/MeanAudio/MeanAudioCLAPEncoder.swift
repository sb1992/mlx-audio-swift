//
//  MeanAudioCLAPEncoder.swift
//  MLXAudio
//
//  LAION CLAP text encoder: RoBERTa-base + linear projection → 512-dim embedding.
//  Used by MeanAudio for text conditioning (text_features_c).
//

import Foundation
import Hub
@preconcurrency import MLX
import MLXAudioCore
import MLXNN
import MLXLMCommon
import Tokenizers

// MARK: - Config

struct CLAPTextConfig: Codable {
    var vocabSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var typeVocabSize: Int
    var layerNormEps: Float
    var hiddenAct: String
    var padTokenId: Int
    var projectionDim: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
        case padTokenId = "pad_token_id"
        case projectionDim = "projection_dim"
    }

    init(
        vocabSize: Int = 50265,
        hiddenSize: Int = 768,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        intermediateSize: Int = 3072,
        maxPositionEmbeddings: Int = 514,
        typeVocabSize: Int = 1,
        layerNormEps: Float = 1e-5,
        hiddenAct: String = "gelu",
        padTokenId: Int = 1,
        projectionDim: Int = 512
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.typeVocabSize = typeVocabSize
        self.layerNormEps = layerNormEps
        self.hiddenAct = hiddenAct
        self.padTokenId = padTokenId
        self.projectionDim = projectionDim
    }
}

// MARK: - RoBERTa Embeddings

fileprivate final class RoBERTaEmbeddings: Module {
    let paddingIdx: Int

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(config: CLAPTextConfig) {
        self.paddingIdx = config.padTokenId
        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize
        )
        self._positionEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize
        )
        self._tokenTypeEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.typeVocabSize, dimensions: config.hiddenSize
        )
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps
        )
        super.init()
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let seqLen = inputIDs.dim(1)

        // RoBERTa position IDs: padding_idx + 1 + arange(seq_len)
        let positionIDs = MLXArray(Int32(paddingIdx + 1) ..< Int32(paddingIdx + 1 + seqLen))
            .expandedDimensions(axis: 0)

        let tokenTypeIDs = MLXArray.zeros(like: inputIDs).asType(.int32)

        var embeddings = wordEmbeddings(inputIDs)
        embeddings = embeddings + positionEmbeddings(positionIDs)
        embeddings = embeddings + tokenTypeEmbeddings(tokenTypeIDs)
        embeddings = layerNorm(embeddings)
        return embeddings
    }
}

// MARK: - RoBERTa Self-Attention

fileprivate final class RoBERTaSelfAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "output") var output: Linear

    init(config: CLAPTextConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        let dim = config.hiddenSize
        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._output.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = hiddenStates.dim(0)
        let S = hiddenStates.dim(1)

        var queries = q(hiddenStates).reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)
        var keys = k(hiddenStates).reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)
        var values = v(hiddenStates).reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        var scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * MLXArray(scale)

        if let mask {
            scores = scores + mask
        }

        let weights = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        var attnOut = matmul(weights, values)
        attnOut = attnOut.transposed(0, 2, 1, 3).reshaped([B, S, numHeads * headDim])
        return output(attnOut)
    }
}

// MARK: - RoBERTa Encoder Layer (post-norm)

fileprivate final class RoBERTaEncoderLayer: Module {
    @ModuleInfo(key: "self_attention") var selfAttention: RoBERTaSelfAttention
    @ModuleInfo(key: "attention_layer_norm") var attentionLayerNorm: LayerNorm
    @ModuleInfo(key: "intermediate") var intermediate: Linear
    @ModuleInfo(key: "output") var outputDense: Linear
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm

    init(config: CLAPTextConfig) {
        self._selfAttention.wrappedValue = RoBERTaSelfAttention(config: config)
        self._attentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps
        )
        self._intermediate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        self._outputDense.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        self._outputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attnOut = selfAttention(hiddenStates, mask: mask)
        let normed1 = attentionLayerNorm(hiddenStates + attnOut)

        let ffOut = outputDense(gelu(intermediate(normed1)))
        return outputLayerNorm(normed1 + ffOut)
    }
}

// MARK: - RoBERTa Model

fileprivate final class RoBERTaModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: RoBERTaEmbeddings
    @ModuleInfo(key: "encoder") var encoder: RoBERTaEncoder
    @ModuleInfo(key: "pooler") var pooler: Linear

    init(config: CLAPTextConfig) {
        self._embeddings.wrappedValue = RoBERTaEmbeddings(config: config)
        self._encoder.wrappedValue = RoBERTaEncoder(config: config)
        self._pooler.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        super.init()
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var extendedMask: MLXArray? = nil
        if let attentionMask {
            var m = attentionMask.asType(.float32)
                .expandedDimensions(axis: 1)
                .expandedDimensions(axis: 1)
            m = (MLXArray(1.0) - m) * MLXArray(-1e9)
            extendedMask = m
        }

        let hidden = embeddings(inputIDs: inputIDs, attentionMask: attentionMask)
        let encoded = encoder(hidden, mask: extendedMask)

        // Pooler: take [CLS] token (index 0), dense + tanh
        let clsToken = encoded[0..., 0, 0...]
        return tanh(pooler(clsToken))
    }
}

fileprivate final class RoBERTaEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [RoBERTaEncoderLayer]

    init(config: CLAPTextConfig) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            RoBERTaEncoderLayer(config: config)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return h
    }
}

// MARK: - CLAP Text Projection

fileprivate final class CLAPTextProjection: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(inputDim: Int, projectionDim: Int) {
        self._linear1.wrappedValue = Linear(inputDim, projectionDim)
        self._linear2.wrappedValue = Linear(projectionDim, projectionDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(relu(linear1(x)))
    }
}

// MARK: - CLAP Text Encoder (public)

fileprivate final class CLAPTextEncoderModel: Module {
    @ModuleInfo(key: "roberta") var roberta: RoBERTaModel
    @ModuleInfo(key: "text_projection") var textProjection: CLAPTextProjection

    init(config: CLAPTextConfig) {
        self._roberta.wrappedValue = RoBERTaModel(config: config)
        self._textProjection.wrappedValue = CLAPTextProjection(
            inputDim: config.hiddenSize, projectionDim: config.projectionDim
        )
        super.init()
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let pooled = roberta(inputIDs: inputIDs, attentionMask: attentionMask)
        let projected = textProjection(pooled)
        // L2 normalize
        let norm = sqrt((projected * projected).sum(axis: -1, keepDims: true))
        return projected / maximum(norm, MLXArray(1e-8))
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("num_batches_tracked") { continue }
            sanitized[key] = value
        }
        return sanitized
    }
}

// MARK: - High-level CLAP Encoder

public final class CLAPTextEncoder: @unchecked Sendable {
    private var model: CLAPTextEncoderModel?
    private var tokenizer: Tokenizers.Tokenizer?
    private let modelFolder: URL
    private let maxLength: Int

    public init(modelFolder: URL, maxLength: Int = 77) {
        self.modelFolder = modelFolder
        self.maxLength = maxLength
    }

    public func ensureLoaded() async throws {
        guard model == nil else { return }

        let configURL = modelFolder.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(CLAPTextConfig.self, from: configData)

        let clapModel = CLAPTextEncoderModel(config: config)
        let rawWeights = try loadCLAPWeights(from: modelFolder)
        let sanitized = CLAPTextEncoderModel.sanitize(weights: rawWeights)
        let params = ModuleParameters.unflattened(sanitized)
        clapModel.update(parameters: params)
        eval(clapModel)

        self.model = clapModel
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    public func encode(_ texts: [String]) async throws -> MLXArray {
        try await ensureLoaded()
        guard let model, let tokenizer else {
            throw CLAPEncoderError.notLoaded
        }

        let tokenized = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }

        var idsFlat: [Int32] = []
        var maskFlat: [Int32] = []
        let batchSize = tokenized.count

        for var ids in tokenized {
            if ids.count > maxLength {
                ids = Array(ids.prefix(maxLength))
            }
            let valid = ids.count
            let padded = ids + Array(repeating: 1, count: max(0, maxLength - valid))

            idsFlat.append(contentsOf: padded.prefix(maxLength).map { Int32($0) })
            maskFlat.append(contentsOf: (0..<maxLength).map { Int32($0 < valid ? 1 : 0) })
        }

        let inputIDs = MLXArray(idsFlat, [batchSize, maxLength])
        let attentionMask = MLXArray(maskFlat, [batchSize, maxLength])

        let embedding = model(inputIDs: inputIDs, attentionMask: attentionMask)
        eval(embedding)
        return embedding
    }

    private func loadCLAPWeights(from folder: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var weights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }
        return weights
    }

    public var isLoaded: Bool { model != nil }
}

enum CLAPEncoderError: Error, LocalizedError {
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .notLoaded: return "CLAP text encoder not loaded"
        }
    }
}
