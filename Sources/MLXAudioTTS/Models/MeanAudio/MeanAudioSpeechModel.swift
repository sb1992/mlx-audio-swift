//
//  MeanAudioSpeechModel.swift
//  MLXAudio
//
//  SpeechGenerationModel conformance for MeanAudio text-to-audio.
//  Handles HF model loading, Python sidecar communication for text encoding,
//  and end-to-end generation: text → T5/CLAP features → flow → VAE → BigVGAN → waveform.
//

import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

// MARK: - Text Encoder Sidecar Client

public struct MeanAudioTextFeatures: Sendable {
    public let textFeatures: MLXArray    // (1, 77, 1024) T5 encoder output
    public let textFeaturesC: MLXArray   // (1, 512) CLAP embedding
}

public enum MeanAudioSidecarError: Error, LocalizedError {
    case sidecarNotRunning(String)
    case encodingFailed(String)
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case .sidecarNotRunning(let msg): return "Text encoder sidecar not running: \(msg)"
        case .encodingFailed(let msg): return "Text encoding failed: \(msg)"
        case .invalidResponse(let msg): return "Invalid sidecar response: \(msg)"
        }
    }
}

public final class MeanAudioTextEncoderClient: Sendable {
    public let baseURL: URL

    public init(host: String = "127.0.0.1", port: Int = 8765) {
        self.baseURL = URL(string: "http://\(host):\(port)")!
    }

    public func encode(text: String) async throws -> MeanAudioTextFeatures {
        let url = baseURL.appendingPathComponent("encode")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60

        let body = try JSONSerialization.data(withJSONObject: ["text": text])
        request.httpBody = body

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw MeanAudioSidecarError.sidecarNotRunning(
                "Cannot connect to text encoder at \(baseURL). Start with: python text_encoder_sidecar.py serve"
            )
        }

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw MeanAudioSidecarError.encodingFailed("HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }

        // Response is an NPZ archive — parse the numpy arrays
        return try parseNpzFeatures(data)
    }

    private func parseNpzFeatures(_ data: Data) throws -> MeanAudioTextFeatures {
        // Write to temp file for MLX to load (MLX can load .npz)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("meanaudio_features_\(UUID().uuidString).npz")
        try data.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let arrays = try MLX.loadArrays(url: tempURL)

        guard let textF = arrays["text_features"],
              let textFC = arrays["text_features_c"] else {
            throw MeanAudioSidecarError.invalidResponse(
                "Expected text_features and text_features_c in response, got: \(Array(arrays.keys))"
            )
        }

        return MeanAudioTextFeatures(
            textFeatures: textF.asType(.float32),
            textFeaturesC: textFC.asType(.float32)
        )
    }

    public func isAvailable() async -> Bool {
        var request = URLRequest(url: baseURL)
        request.httpMethod = "HEAD"
        request.timeoutInterval = 2
        return (try? await URLSession.shared.data(for: request)) != nil
    }
}

// MARK: - MeanAudio Speech Model

public final class MeanAudioModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: MeanAudioConfig
    private let pipeline: MeanAudioPipeline
    private let textEncoder: MeanAudioTextEncoderClient
    private let modelFolder: URL

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 0.0)
    }

    private init(config: MeanAudioConfig, pipeline: MeanAudioPipeline, textEncoder: MeanAudioTextEncoderClient, modelFolder: URL) {
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

        // Encode text via Python sidecar
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

        // Capture all params as sendable values before Task
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

    public var parameterCount: Int { pipeline.parameterCount }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        sidecarHost: String = "127.0.0.1",
        sidecarPort: Int = 8765,
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

        return try await fromModelDirectory(
            modelDir,
            sidecarHost: sidecarHost,
            sidecarPort: sidecarPort
        )
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        sidecarHost: String = "127.0.0.1",
        sidecarPort: Int = 8765
    ) async throws -> MeanAudioModel {
        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MeanAudioConfig.self, from: configData)

        let pipeline = MeanAudioPipeline(config: config)

        // Load flow transformer weights
        let flowURL = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: flowURL.path) {
            try pipeline.loadFlowWeights(from: flowURL)
        }

        // Load VAE weights
        let vaeURL = modelDir.appendingPathComponent("vae.safetensors")
        if FileManager.default.fileExists(atPath: vaeURL.path) {
            try pipeline.loadVAEWeights(from: vaeURL)
        }

        // Load BigVGAN vocoder
        let vocoderWeightsURL = modelDir.appendingPathComponent("bigvgan.safetensors")
        let vocoderConfigURL = modelDir.appendingPathComponent("bigvgan_config.json")
        if FileManager.default.fileExists(atPath: vocoderWeightsURL.path),
           FileManager.default.fileExists(atPath: vocoderConfigURL.path) {
            let vocoderConfigData = try Data(contentsOf: vocoderConfigURL)
            let vocoderConfig = try JSONDecoder().decode(BigVGANConfig.self, from: vocoderConfigData)
            try pipeline.loadVocoderWeights(config: vocoderConfig, from: vocoderWeightsURL)
        }

        let textEncoder = MeanAudioTextEncoderClient(host: sidecarHost, port: sidecarPort)

        return MeanAudioModel(
            config: config,
            pipeline: pipeline,
            textEncoder: textEncoder,
            modelFolder: modelDir
        )
    }
}
