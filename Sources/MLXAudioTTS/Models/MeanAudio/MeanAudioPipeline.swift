//
//  MeanAudioPipeline.swift
//  MLXAudio
//
//  End-to-end MeanAudio inference pipeline:
//  text features → flow transformer → VAE decoder → BigVGAN → waveform
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon

public struct MeanAudioGenerateOptions: Sendable {
    public var cfgStrength: Float
    public var steps: Int
    public var seed: UInt64?

    public init(cfgStrength: Float = 4.5, steps: Int = 1, seed: UInt64? = nil) {
        self.cfgStrength = cfgStrength
        self.steps = steps
        self.seed = seed
    }
}

public final class MeanAudioPipeline {
    public let config: MeanAudioConfig
    let flowModel: MeanAudioFlowTransformer
    let vae: MeanAudioVAE
    let sampler: MeanFlowSampler

    public init(config: MeanAudioConfig) {
        self.config = config
        self.flowModel = MeanAudioFlowTransformer(config: config)
        self.vae = MeanAudioVAE(dataDim: 80, embedDim: config.latentDim, hiddenDim: 384)
        self.sampler = MeanFlowSampler(steps: config.steps)
    }

    public func loadFlowWeights(from url: URL) throws {
        let weights = try loadArrays(url: url)
        let params = ModuleParameters.unflattened(weights)
        flowModel.update(parameters: params)
        eval(flowModel)
    }

    public func loadVAEWeights(from url: URL) throws {
        let weights = try loadArrays(url: url)
        let params = ModuleParameters.unflattened(weights)
        vae.update(parameters: params)
        eval(vae)
    }

    /// Generate audio from pre-computed text features.
    ///
    /// - Parameters:
    ///   - textFeatures: (B, 77, 1024) T5 encoder output
    ///   - textFeaturesC: (B, 512) CLAP embedding
    ///   - options: generation options (CFG strength, steps, seed)
    /// - Returns: Mel spectrogram (B, T, 80) ready for BigVGAN vocoding
    public func generateMel(
        textFeatures: MLXArray,
        textFeaturesC: MLXArray,
        options: MeanAudioGenerateOptions = .init()
    ) -> MLXArray {
        let bs = textFeatures.dim(0)

        // Seed RNG
        if let seed = options.seed {
            MLXRandom.seed(seed)
        }

        // 1. Preprocess text conditions
        let conditions = flowModel.preprocessConditions(
            textF: textFeatures, textFC: textFeaturesC
        )
        let emptyConditions = flowModel.getEmptyConditions(batchSize: bs)
        eval(conditions.textF, conditions.textFC, emptyConditions.textF, emptyConditions.textFC)

        // 2. Sample noise
        let noise = MLXRandom.normal([bs, config.latentSeqLen, config.latentDim])

        // 3. Run ODE sampler
        let sampler = MeanFlowSampler(steps: options.steps)
        let latent = sampler.sample(
            model: flowModel,
            noise: noise,
            conditions: conditions,
            emptyConditions: emptyConditions,
            cfgStrength: options.cfgStrength
        )

        // 4. Unnormalize latent
        let unnormed = flowModel.unnormalize(latent)
        eval(unnormed)

        // 5. VAE decode: latent (B, Seq, D) → transpose to (B, D, Seq) → mel (B, C, T)
        let latentChannelsFirst = unnormed.transposed(0, 2, 1)
        let melChannelsFirst = vae.decode(latentChannelsFirst)
        eval(melChannelsFirst)

        // 6. Return mel in (B, T, C) for BigVGAN
        return melChannelsFirst.transposed(0, 2, 1)
    }

    /// Full parameter count for logging.
    public var parameterCount: Int {
        let flowParams = flowModel.parameters().flattened().map(\.1.size).reduce(0, +)
        let vaeParams = vae.parameters().flattened().map(\.1.size).reduce(0, +)
        return flowParams + vaeParams
    }
}

// MARK: - Weight Loading Helpers

func loadArrays(url: URL) throws -> [String: MLXArray] {
    if url.pathExtension == "safetensors" {
        return try MLX.loadArrays(url: url)
    } else {
        return try MLX.loadArrays(url: url)
    }
}
