//
//  MeanAudioSpeechModel.swift
//  MLXAudio
//
//  SpeechGenerationModel conformance for MeanAudio text-to-audio.
//  End-to-end generation: text → T5/CLAP features → flow → VAE → BigVGAN → waveform.
//

import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

// MARK: - Text Features

public struct MeanAudioTextFeatures: Sendable {
    public let textFeatures: MLXArray    // (1, 77, 1024) T5 encoder output
    public let textFeaturesC: MLXArray   // (1, 512) CLAP embedding
}

// MARK: - Native Text Encoder

public final class MeanAudioNativeTextEncoder: @unchecked Sendable {
    private let t5Encoder: MeanAudioT5Encoder
    private let clapEncoder: CLAPTextEncoder

    public init(t5Folder: URL, clapFolder: URL, maxLength: Int = 77) {
        self.t5Encoder = MeanAudioT5Encoder(modelFolder: t5Folder, maxLength: maxLength)
        self.clapEncoder = CLAPTextEncoder(modelFolder: clapFolder, maxLength: maxLength)
    }

    public func ensureLoaded() async throws {
        try await t5Encoder.ensureLoaded()
        try await clapEncoder.ensureLoaded()
    }

    public func encode(text: String) async throws -> MeanAudioTextFeatures {
        let t5Features = try await t5Encoder.encode([text])
        let clapFeatures = try await clapEncoder.encode([text])

        return MeanAudioTextFeatures(
            textFeatures: t5Features.asType(.float32),
            textFeaturesC: clapFeatures.asType(.float32)
        )
    }

    public var isLoaded: Bool { t5Encoder.isLoaded && clapEncoder.isLoaded }
}

// MARK: - MeanAudio Speech Model

public final class MeanAudioModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: MeanAudioConfig
    private let pipeline: MeanAudioPipeline
    private let textEncoder: MeanAudioNativeTextEncoder
    private let modelFolder: URL

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 0.0)
    }

    private init(
        config: MeanAudioConfig,
        pipeline: MeanAudioPipeline,
        textEncoder: MeanAudioNativeTextEncoder,
        modelFolder: URL
    ) {
        self.config = config
        self.pipeline = pipeline
        self.textEncoder = textEncoder
        self.modelFolder = modelFolder
        super.init()
    }

    // MARK: - SpeechGenerationModel

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        try Task.checkCancellation()

        let features = try await textEncoder.encode(text: text)

        try Task.checkCancellation()

        let options = MeanAudioGenerateOptions(
            cfgStrength: config.cfgStrength,
            steps: config.steps,
            seed: nil
        )

        return try pipeline.generateAudio(
            textFeatures: features.textFeatures,
            textFeaturesC: features.textFeaturesC,
            options: options
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        let capturedText = text
        let capturedVoice = voice
        let capturedRefAudio = refAudio
        let capturedRefText = refText
        let capturedLanguage = language
        let capturedParams = generationParameters

        let task = Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                let audio = try await self.generate(
                    text: capturedText,
                    voice: capturedVoice,
                    refAudio: capturedRefAudio,
                    refText: capturedRefText,
                    language: capturedLanguage,
                    generationParameters: capturedParams
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }

        return stream
    }

    // MARK: - Direct API (without SpeechGenerationModel)

    public func generateFromFeatures(
        textFeatures: MLXArray,
        textFeaturesC: MLXArray,
        options: MeanAudioGenerateOptions = .init()
    ) throws -> MLXArray {
        try pipeline.generateAudio(
            textFeatures: textFeatures,
            textFeaturesC: textFeaturesC,
            options: options
        )
    }

    public func generateMelFromFeatures(
        textFeatures: MLXArray,
        textFeaturesC: MLXArray,
        options: MeanAudioGenerateOptions = .init()
    ) -> MLXArray {
        pipeline.generateMel(
            textFeatures: textFeatures,
            textFeaturesC: textFeaturesC,
            options: options
        )
    }

    public func encodeText(_ text: String) async throws -> MeanAudioTextFeatures {
        try await textEncoder.encode(text: text)
    }

    public var parameterCount: Int { pipeline.parameterCount }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> MeanAudioModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw TTSModelError.invalidRepositoryID(modelRepo)
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            hfToken: hfToken,
            cache: cache
        )

        return try fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) throws -> MeanAudioModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MeanAudioConfig.self, from: configData)

        let pipeline = MeanAudioPipeline(config: config)

        // Load flow transformer
        let flowURL = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: flowURL.path) {
            try pipeline.loadFlowWeights(from: flowURL)
        }

        // Load VAE
        let vaeURL = modelDir.appendingPathComponent("vae.safetensors")
        if FileManager.default.fileExists(atPath: vaeURL.path) {
            try pipeline.loadVAEWeights(from: vaeURL)
        }

        // Load BigVGAN
        let vocoderWeightsURL = modelDir.appendingPathComponent("bigvgan.safetensors")
        let vocoderConfigURL = modelDir.appendingPathComponent("bigvgan_config.json")
        if FileManager.default.fileExists(atPath: vocoderWeightsURL.path),
           FileManager.default.fileExists(atPath: vocoderConfigURL.path) {
            let vocoderConfigData = try Data(contentsOf: vocoderConfigURL)
            let vocoderConfig = try JSONDecoder().decode(BigVGANConfig.self, from: vocoderConfigData)
            try pipeline.loadVocoderWeights(config: vocoderConfig, from: vocoderWeightsURL)
        }

        // Text encoders: look for t5/ and clap/ subdirectories
        let t5Folder = modelDir.appendingPathComponent("t5")
        let clapFolder = modelDir.appendingPathComponent("clap")

        let textEncoder = MeanAudioNativeTextEncoder(
            t5Folder: t5Folder,
            clapFolder: clapFolder,
            maxLength: config.textSeqLen
        )

        return MeanAudioModel(
            config: config,
            pipeline: pipeline,
            textEncoder: textEncoder,
            modelFolder: modelDir
        )
    }
}
