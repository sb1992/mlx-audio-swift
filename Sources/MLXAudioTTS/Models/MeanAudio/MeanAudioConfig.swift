//
//  MeanAudioConfig.swift
//  MLXAudio
//
//  Configuration for MeanAudio text-to-audio model.
//  Ported from MeanAudio Python: meanaudio/model/networks.py + sequence_config.py
//

import Foundation

public struct MeanAudioConfig: Codable, Sendable {
    public var latentDim: Int
    public var textDim: Int
    public var textCDim: Int
    public var hiddenDim: Int
    public var depth: Int
    public var fusedDepth: Int
    public var numHeads: Int
    public var mlpRatio: Float
    public var latentSeqLen: Int
    public var textSeqLen: Int
    public var useRope: Bool
    public var sampleRate: Int
    public var durationSeconds: Float
    public var cfgStrength: Float
    public var steps: Int

    enum CodingKeys: String, CodingKey {
        case latentDim = "latent_dim"
        case textDim = "text_dim"
        case textCDim = "text_c_dim"
        case hiddenDim = "hidden_dim"
        case depth
        case fusedDepth = "fused_depth"
        case numHeads = "num_heads"
        case mlpRatio = "mlp_ratio"
        case latentSeqLen = "latent_seq_len"
        case textSeqLen = "text_seq_len"
        case useRope = "use_rope"
        case sampleRate = "sample_rate"
        case durationSeconds = "duration_seconds"
        case cfgStrength = "cfg_strength"
        case steps
    }

    public init(
        latentDim: Int = 20,
        textDim: Int = 1024,
        textCDim: Int = 512,
        hiddenDim: Int = 448,
        depth: Int = 12,
        fusedDepth: Int = 8,
        numHeads: Int = 7,
        mlpRatio: Float = 4.0,
        latentSeqLen: Int = 312,
        textSeqLen: Int = 77,
        useRope: Bool = false,
        sampleRate: Int = 16000,
        durationSeconds: Float = 9.975,
        cfgStrength: Float = 4.5,
        steps: Int = 1
    ) {
        self.latentDim = latentDim
        self.textDim = textDim
        self.textCDim = textCDim
        self.hiddenDim = hiddenDim
        self.depth = depth
        self.fusedDepth = fusedDepth
        self.numHeads = numHeads
        self.mlpRatio = mlpRatio
        self.latentSeqLen = latentSeqLen
        self.textSeqLen = textSeqLen
        self.useRope = useRope
        self.sampleRate = sampleRate
        self.durationSeconds = durationSeconds
        self.cfgStrength = cfgStrength
        self.steps = steps
    }

    public static let small = MeanAudioConfig(
        latentDim: 20,
        textDim: 1024,
        textCDim: 512,
        hiddenDim: 64 * 7,
        depth: 12,
        fusedDepth: 8,
        numHeads: 7,
        latentSeqLen: 312
    )

    public static let large = MeanAudioConfig(
        latentDim: 20,
        textDim: 1024,
        textCDim: 512,
        hiddenDim: 64 * 14,
        depth: 12,
        fusedDepth: 8,
        numHeads: 14,
        latentSeqLen: 312
    )

    public var headDim: Int { hiddenDim / numHeads }
    public var mmDepth: Int { depth - fusedDepth }
}
