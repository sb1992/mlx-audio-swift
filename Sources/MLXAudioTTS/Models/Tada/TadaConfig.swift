import Foundation
import MLX
import MLXNN

// MARK: - Model Config (from config.json)

public struct TadaConfig: Codable, Sendable {
    public var vocabSize: Int = 128256
    public var hiddenSize: Int = 2048
    public var intermediateSize: Int = 8192
    public var numHiddenLayers: Int = 16
    public var numAttentionHeads: Int = 32
    public var numKeyValueHeads: Int = 8
    public var headDim: Int = 64
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 500000.0
    public var ropeScaling: TadaRopeScaling = .init()
    public var tieWordEmbeddings: Bool = true
    public var maxPositionEmbeddings: Int = 131072
    public var acousticDim: Int = 512
    public var numTimeClasses: Int = 256
    public var shiftAcoustic: Int = 5
    public var headLayers: Int = 6
    public var headFfnRatio: Float = 4.0
    public var bottleneckDim: Int?
    public var acousticMean: Float = 0.0
    public var acousticStd: Float = 1.5
    public var bosTokenId: Int = 128000
    public var eosTokenId: [Int] = [128001, 128008, 128009]
    public var padTokenId: Int = 128004
    public var startHeaderId: Int = 128006
    public var endHeaderId: Int = 128007
    public var eotId: Int = 128009

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case acousticDim = "acoustic_dim"
        case numTimeClasses = "num_time_classes"
        case shiftAcoustic = "shift_acoustic"
        case headLayers = "head_layers"
        case headFfnRatio = "head_ffn_ratio"
        case bottleneckDim = "bottleneck_dim"
        case acousticMean = "acoustic_mean"
        case acousticStd = "acoustic_std"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case startHeaderId = "start_header_id"
        case endHeaderId = "end_header_id"
        case eotId = "eot_id"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128256
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500000.0
        ropeScaling = try c.decodeIfPresent(TadaRopeScaling.self, forKey: .ropeScaling) ?? .init()
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        acousticDim = try c.decodeIfPresent(Int.self, forKey: .acousticDim) ?? 512
        numTimeClasses = try c.decodeIfPresent(Int.self, forKey: .numTimeClasses) ?? 256
        shiftAcoustic = try c.decodeIfPresent(Int.self, forKey: .shiftAcoustic) ?? 5
        headLayers = try c.decodeIfPresent(Int.self, forKey: .headLayers) ?? 6
        headFfnRatio = try c.decodeIfPresent(Float.self, forKey: .headFfnRatio) ?? 4.0
        bottleneckDim = try c.decodeIfPresent(Int.self, forKey: .bottleneckDim)
        acousticMean = try c.decodeIfPresent(Float.self, forKey: .acousticMean) ?? 0.0
        acousticStd = try c.decodeIfPresent(Float.self, forKey: .acousticStd) ?? 1.5
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 128000
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 128004
        startHeaderId = try c.decodeIfPresent(Int.self, forKey: .startHeaderId) ?? 128006
        endHeaderId = try c.decodeIfPresent(Int.self, forKey: .endHeaderId) ?? 128007
        eotId = try c.decodeIfPresent(Int.self, forKey: .eotId) ?? 128009

        if let single = try? c.decode(Int.self, forKey: .eosTokenId) {
            eosTokenId = [single]
        } else {
            eosTokenId = try c.decodeIfPresent([Int].self, forKey: .eosTokenId) ?? [128001, 128008, 128009]
        }
    }
}

// MARK: - RoPE Scaling

public struct TadaRopeScaling: Codable, Sendable {
    public var factor: Float = 32.0
    public var highFreqFactor: Float = 4.0
    public var lowFreqFactor: Float = 1.0
    public var originalMaxPositionEmbeddings: Int = 8192
    public var ropeType: String = "llama3"

    enum CodingKeys: String, CodingKey {
        case factor
        case highFreqFactor = "high_freq_factor"
        case lowFreqFactor = "low_freq_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case ropeType = "rope_type"
    }

    public init() {}
}

// MARK: - Inference Options

public struct TadaInferenceOptions: Sendable {
    public var textDoSample: Bool = true
    public var textTemperature: Float = 0.6
    public var textTopK: Int = 0
    public var textTopP: Float = 0.9
    public var textRepetitionPenalty: Float = 1.1
    public var acousticCfgScale: Float = 1.6
    public var durationCfgScale: Float = 1.0
    public var cfgSchedule: CfgSchedule = .cosine
    public var noiseTemperature: Float = 0.9
    public var numFlowMatchingSteps: Int = 20
    public var timeSchedule: TimeSchedule = .logsnr

    public enum CfgSchedule: String, Sendable {
        case constant, linear, cosine
    }

    public enum TimeSchedule: String, Sendable {
        case uniform, cosine, logsnr
    }

    public init() {}
}

// MARK: - Pre-encoded Reference

public struct TadaReference: @unchecked Sendable {
    public let tokenValues: MLXArray
    public let tokenPositions: MLXArray
    public let tokenMasks: MLXArray?
    public let textTokens: MLXArray
    public let textTokensLen: MLXArray
    public let audioLen: MLXArray
    public let text: String
    public let sampleRate: Int

    public init(
        tokenValues: MLXArray,
        tokenPositions: MLXArray,
        tokenMasks: MLXArray?,
        textTokens: MLXArray,
        textTokensLen: MLXArray,
        audioLen: MLXArray,
        text: String,
        sampleRate: Int = 24000
    ) {
        self.tokenValues = tokenValues
        self.tokenPositions = tokenPositions
        self.tokenMasks = tokenMasks
        self.textTokens = textTokens
        self.textTokensLen = textTokensLen
        self.audioLen = audioLen
        self.text = text
        self.sampleRate = sampleRate
    }
}
