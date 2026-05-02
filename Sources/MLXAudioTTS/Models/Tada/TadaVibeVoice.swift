import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Timestep Embedder

class TadaTimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int
    @ModuleInfo(key: "mlp_0") var mlp0: Linear
    @ModuleInfo(key: "mlp_2") var mlp2: Linear

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self._mlp0.wrappedValue = Linear(frequencyEmbeddingSize, hiddenSize, bias: false)
        self._mlp2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    }

    static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000) -> MLXArray {
        let half = dim / 2
        let freqs = exp(
            -log(Float(maxPeriod)) * MLXArray(0..<half).asType(.float32) / Float(half)
        )
        let args = expandedDimensions(t.asType(.float32), axis: 1) * expandedDimensions(freqs, axis: 0)
        var embedding = concatenated([cos(args), sin(args)], axis: -1)
        if dim % 2 != 0 {
            embedding = concatenated([embedding, MLXArray.zeros(like: embedding[0..., ..<1])], axis: -1)
        }
        return embedding.asType(t.dtype)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize)
        return mlp2(silu(mlp0(tFreq)))
    }
}

// MARK: - SiLU-gated FFN

class TadaFFN: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(embedDim: Int, ffnDim: Int) {
        self._gateProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        self._upProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        self._downProj.wrappedValue = Linear(ffnDim, embedDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - AdaLN modulation

private func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1.0 + scale) + shift
}

// MARK: - Head Layer (AdaLN-modulated FFN)

class TadaHeadLayer: Module {
    var ffn: TadaFFN
    var norm: RMSNorm
    @ModuleInfo(key: "adaLN_modulation_linear") var adaLNModulationLinear: Linear

    init(embedDim: Int, ffnDim: Int, condDim: Int, normEps: Float = 1e-5) {
        self.ffn = TadaFFN(embedDim: embedDim, ffnDim: ffnDim)
        self.norm = RMSNorm(dimensions: embedDim, eps: normEps)
        self._adaLNModulationLinear.wrappedValue = Linear(condDim, 3 * embedDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let mod = adaLNModulationLinear(silu(c))
        let parts = split(mod, parts: 3, axis: -1)
        let (shift, scale, gate) = (parts[0], parts[1], parts[2])
        return x + gate * ffn(modulate(norm(x), shift: shift, scale: scale))
    }
}

// MARK: - Final Layer

class TadaFinalLayer: Module {
    @ModuleInfo(key: "norm_final") var normFinal: RMSNorm
    var linear: Linear
    @ModuleInfo(key: "adaLN_modulation_linear") var adaLNModulationLinear: Linear

    init(hiddenSize: Int, outputSize: Int, condSize: Int, normEps: Float = 1e-5) {
        self._normFinal.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: normEps)
        self.linear = Linear(hiddenSize, outputSize, bias: false)
        self._adaLNModulationLinear.wrappedValue = Linear(condSize, 2 * hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let mod = adaLNModulationLinear(silu(c))
        let parts = split(mod, parts: 2, axis: -1)
        return linear(modulate(normFinal(x), shift: parts[0], scale: parts[1]))
    }
}

// MARK: - VibeVoice Diffusion Head

class TadaVibeVoiceDiffusionHead: Module {
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "noisy_images_proj") var noisyImagesProj: Linear
    @ModuleInfo(key: "t_embedder") var tEmbedder: TadaTimestepEmbedder
    var layers: [TadaHeadLayer]
    @ModuleInfo(key: "final_layer") var finalLayer: TadaFinalLayer

    let hiddenSize: Int
    let latentSize: Int

    init(hiddenSize: Int, headLayers: Int, headFfnRatio: Float, latentSize: Int, rmsNormEps: Float) {
        self.hiddenSize = hiddenSize
        self.latentSize = latentSize
        let condDim = hiddenSize
        let ffnDim = Int(Float(hiddenSize) * headFfnRatio)

        self._condProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._noisyImagesProj.wrappedValue = Linear(latentSize, hiddenSize, bias: false)
        self._tEmbedder.wrappedValue = TadaTimestepEmbedder(hiddenSize: condDim)
        self.layers = (0..<headLayers).map { _ in
            TadaHeadLayer(embedDim: hiddenSize, ffnDim: ffnDim, condDim: condDim, normEps: rmsNormEps)
        }
        self._finalLayer.wrappedValue = TadaFinalLayer(
            hiddenSize: hiddenSize, outputSize: latentSize, condSize: condDim, normEps: rmsNormEps
        )
    }

    func callAsFunction(noisyImages: MLXArray, timesteps: MLXArray, condition: MLXArray) -> MLXArray {
        var x = noisyImagesProj(noisyImages)
        let t = tEmbedder(timesteps)
        let conditioning = condProj(condition) + t
        for layer in layers {
            x = layer(x, c: conditioning)
        }
        return finalLayer(x, c: conditioning)
    }

    // MARK: - CFG Scheduling

    static func scheduledCfg(baseScale: Float, t: Float, schedule: TadaInferenceOptions.CfgSchedule) -> Float {
        if schedule == .constant || baseScale == 1.0 { return baseScale }
        switch schedule {
        case .linear: return 1.0 + (baseScale - 1.0) * (1.0 - t)
        case .cosine: return 1.0 + (baseScale - 1.0) * 0.5 * (1.0 + Foundation.cos(Float.pi * t))
        case .constant: return baseScale
        }
    }

    // MARK: - Time Schedule

    static func buildTimeSchedule(numSteps: Int, schedule: TadaInferenceOptions.TimeSchedule) -> MLXArray {
        switch schedule {
        case .cosine:
            let linear = MLXArray.linspace(Float(0), Float(1), count: numSteps + 1)
            return 0.5 * (1.0 - MLX.cos(Float.pi * linear))
        case .logsnr:
            let logSnr = MLXArray.linspace(Float(5.0), Float(-5.0), count: numSteps + 1)
            var tSpan = sigmoid(-logSnr / 2.0)
            let first = tSpan[0]
            let last = tSpan[numSteps]
            tSpan = tSpan - first
            let scale = 1.0 / (last - first)
            tSpan = tSpan * scale
            return tSpan
        case .uniform:
            return MLXArray.linspace(Float(0), Float(1), count: numSteps + 1)
        }
    }

    // MARK: - Velocity with CFG

    func computeVelocity(
        speech: MLXArray, t: MLXArray,
        cond: MLXArray, negCond: MLXArray,
        acousticDim: Int,
        acousticCfg: Float, durationCfg: Float,
        bottleneckFn: ((MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        let applyBn = bottleneckFn ?? { $0 }

        if acousticCfg != 1.0 {
            let speechComb = concatenated([speech, speech], axis: 0)
            let tComb = tiled(t, repetitions: [speech.shape[0] * 2])
            let condPos = cond.ndim == 3 ? squeezed(cond, axis: 1) : cond
            let condNeg = negCond.ndim == 3 ? squeezed(negCond, axis: 1) : negCond
            let condComb = concatenated([condPos, condNeg], axis: 0)
            let velComb = callAsFunction(noisyImages: speechComb, timesteps: tComb, condition: applyBn(condComb))
            let parts = split(velComb, parts: 2, axis: 0)
            let velPos = parts[0]
            let velNeg = parts[1]
            let velAcoustic = velNeg[.ellipsis, ..<acousticDim] + acousticCfg * (
                velPos[.ellipsis, ..<acousticDim] - velNeg[.ellipsis, ..<acousticDim]
            )
            let velTime = velNeg[.ellipsis, acousticDim...] + durationCfg * (
                velPos[.ellipsis, acousticDim...] - velNeg[.ellipsis, acousticDim...]
            )
            return concatenated([velAcoustic, velTime], axis: -1)
        }

        let condSq = cond.ndim == 3 ? squeezed(cond, axis: 1) : cond
        return callAsFunction(
            noisyImages: speech,
            timesteps: tiled(t, repetitions: [speech.shape[0]]),
            condition: applyBn(condSq)
        )
    }

    // MARK: - Euler ODE Solver

    func solve(
        noise: MLXArray,
        cond: MLXArray,
        negCond: MLXArray,
        acousticDim: Int,
        numSteps: Int,
        acousticCfgScale: Float,
        durationCfgScale: Float,
        cfgSchedule: TadaInferenceOptions.CfgSchedule = .constant,
        timeSchedule: TadaInferenceOptions.TimeSchedule = .uniform,
        bottleneckFn: ((MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        let origDtype = noise.dtype
        var speech = noise.asType(.float32)
        let condF = cond.asType(.float32)
        let negCondF = negCond.asType(.float32)

        let tSpanMx = Self.buildTimeSchedule(numSteps: numSteps, schedule: timeSchedule)
        eval(tSpanMx)

        var tSpan = [Float]()
        for i in 0...numSteps {
            tSpan.append(tSpanMx[i].item(Float.self))
        }

        for i in 1..<tSpan.count {
            let dt = tSpan[i] - tSpan[i - 1]
            let tVal = tSpan[i - 1]
            let tMx = MLXArray(tVal)
            let aCfg = Self.scheduledCfg(baseScale: acousticCfgScale, t: tVal, schedule: cfgSchedule)
            let dCfg = Self.scheduledCfg(baseScale: durationCfgScale, t: tVal, schedule: cfgSchedule)
            let velocity = computeVelocity(
                speech: speech, t: tMx, cond: condF, negCond: negCondF,
                acousticDim: acousticDim, acousticCfg: aCfg, durationCfg: dCfg,
                bottleneckFn: bottleneckFn
            )
            speech = speech + dt * velocity
        }

        return speech.asType(origDtype)
    }
}
