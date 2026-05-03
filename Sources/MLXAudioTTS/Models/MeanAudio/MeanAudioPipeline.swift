//
//  MeanAudioPipeline.swift
//  MLXAudio
//
//  End-to-end MeanAudio inference pipeline:
//  text features → flow transformer → VAE decoder → BigVGAN → waveform
//

import Foundation
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
import MLXNN
import MLXLMCommon

public struct MeanAudioGenerateOptions: Sendable {
    public var cfgStrength: Float
    public var steps: Int
    public var seed: UInt64?

    public init(cfgStrength: Float = 0, steps: Int = 1, seed: UInt64? = nil) {
        self.cfgStrength = cfgStrength
        self.steps = steps
        self.seed = seed
    }
}

public final class MeanAudioPipeline {
    public let config: MeanAudioConfig
    let flowModel: MeanAudioFlowTransformer
    let vae: MeanAudioVAE
    public var vocoder: BigVGAN?

    public init(config: MeanAudioConfig) {
        self.config = config
        self.flowModel = MeanAudioFlowTransformer(config: config)
        self.vae = MeanAudioVAE(dataDim: 80, embedDim: config.latentDim, hiddenDim: 384)
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

    public func loadVocoderWeights(config vocoderConfig: BigVGANConfig, from url: URL) throws {
        let bigvgan = BigVGAN(config: vocoderConfig)
        let rawWeights = try MLX.loadArrays(url: url)
        let sanitized = bigvgan.sanitize(weights: rawWeights)
        let params = ModuleParameters.unflattened(sanitized)
        bigvgan.update(parameters: params)
        eval(bigvgan)
        self.vocoder = bigvgan
    }

    /// Generate mel spectrogram from pre-computed text features.
    public func generateMel(
        textFeatures: MLXArray,
        textFeaturesC: MLXArray,
        options: MeanAudioGenerateOptions = .init()
    ) -> MLXArray {
        let bs = textFeatures.dim(0)

        if let seed = options.seed {
            MLXRandom.seed(seed)
        }

        let conditions = flowModel.preprocessConditions(
            textF: textFeatures, textFC: textFeaturesC
        )
        let emptyConditions = flowModel.getEmptyConditions(batchSize: bs)
        eval(conditions.textF, conditions.textFC, emptyConditions.textF, emptyConditions.textFC)

        let noise = MLXRandom.normal([bs, config.latentSeqLen, config.latentDim])
        eval(noise)

        let sampler = MeanFlowSampler(steps: options.steps)
        let latent = sampler.sample(
            model: flowModel,
            noise: noise,
            conditions: conditions,
            emptyConditions: emptyConditions,
            cfgStrength: options.cfgStrength
        )
        eval(latent)

        let unnormed = flowModel.unnormalize(latent)
        let latentCF = unnormed.transposed(0, 2, 1)
        let mel = vae.decode(latentCF)
        eval(mel)
        return mel
    }

    /// Generate waveform from pre-computed text features (full pipeline: flow → VAE → BigVGAN).
    public func generateAudio(
        textFeatures: MLXArray,
        textFeaturesC: MLXArray,
        options: MeanAudioGenerateOptions = .init()
    ) throws -> MLXArray {
        guard let vocoder else {
            throw AudioGenerationError.modelNotInitialized("BigVGAN vocoder not loaded")
        }

        let mel = generateMel(
            textFeatures: textFeatures,
            textFeaturesC: textFeaturesC,
            options: options
        )

        let waveform = vocoder(mel)
        eval(waveform)

        // Squeeze to (T_audio,) for single-batch
        return waveform.squeezed()
    }

    public func decodeLatent(_ normalizedLatent: MLXArray) throws -> MLXArray {
        guard let vocoder else {
            throw AudioGenerationError.modelNotInitialized("BigVGAN vocoder not loaded")
        }
        let unnormed = flowModel.unnormalize(normalizedLatent)
        let latentCF = unnormed.transposed(0, 2, 1)
        let mel = vae.decode(latentCF)
        let waveform = vocoder(mel)
        eval(waveform)
        return waveform.squeezed()
    }

    public func decodeVAE(_ channelsFirstLatent: MLXArray) -> MLXArray {
        vae.decode(channelsFirstLatent)
    }

    public func vocode(_ mel: MLXArray) throws -> MLXArray {
        guard let vocoder else {
            throw AudioGenerationError.modelNotInitialized("BigVGAN vocoder not loaded")
        }
        let waveform = vocoder(mel)
        eval(waveform)
        return waveform.squeezed()
    }

    public var latentMean: MLXArray { flowModel.latentMean }
    public var latentStd: MLXArray { flowModel.latentStd }

    public var parameterCount: Int {
        let flowParams = flowModel.parameters().flattened().map(\.1.size).reduce(0, +)
        let vaeParams = vae.parameters().flattened().map(\.1.size).reduce(0, +)
        let vocoderParams = vocoder?.parameters().flattened().map(\.1.size).reduce(0, +) ?? 0
        return flowParams + vaeParams + vocoderParams
    }

}

// MARK: - Weight Loading Helpers

func loadArrays(url: URL) throws -> [String: MLXArray] {
    try MLX.loadArrays(url: url)
}
