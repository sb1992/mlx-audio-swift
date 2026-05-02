import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
@preconcurrency import MLXLMCommon
import HuggingFace
import Tokenizers

// MARK: - TadaForCausalLM (core inference module)

class TadaForCausalLM: Module {
    let config: TadaConfig
    let numTimeBits: Int
    let timeDim: Int

    var model: TadaLlamaModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "acoustic_proj") var acousticProj: Linear
    @ModuleInfo(key: "time_start_embed") var timeStartEmbed: Embedding
    @ModuleInfo(key: "time_end_embed") var timeEndEmbed: Embedding
    @ModuleInfo(key: "acoustic_mask_emb") var acousticMaskEmb: Embedding
    var bottleneckProj: Linear?
    @ModuleInfo(key: "prediction_head") var predictionHead: TadaVibeVoiceDiffusionHead

    var encoder: TadaEncoder?
    var decoder: TadaDecoder?
    var aligner: TadaAligner?
    var tokenizer: Tokenizers.Tokenizer?

    private var ropeCos: MLXArray?
    private var ropeSin: MLXArray?
    private var ropeLen: Int = 0

    init(config: TadaConfig) {
        self.config = config
        self.numTimeBits = Int(ceil(log2(Double(config.numTimeClasses))))
        self.timeDim = 2 * numTimeBits

        self.model = TadaLlamaModel(config: config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }

        self._acousticProj.wrappedValue = Linear(config.acousticDim, config.hiddenSize)
        self._timeStartEmbed.wrappedValue = Embedding(embeddingCount: config.numTimeClasses, dimensions: config.hiddenSize)
        self._timeEndEmbed.wrappedValue = Embedding(embeddingCount: config.numTimeClasses, dimensions: config.hiddenSize)
        self._acousticMaskEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.hiddenSize)

        if let bd = config.bottleneckDim {
            self.bottleneckProj = Linear(config.hiddenSize, bd)
        }

        let headHidden = config.bottleneckDim ?? config.hiddenSize
        let latentSize = config.acousticDim + timeDim
        self._predictionHead.wrappedValue = TadaVibeVoiceDiffusionHead(
            hiddenSize: headHidden,
            headLayers: config.headLayers,
            headFfnRatio: config.headFfnRatio,
            latentSize: latentSize,
            rmsNormEps: 1e-5
        )
    }

    // MARK: - RoPE

    func ensureRope(seqLen: Int) {
        guard seqLen > ropeLen else { return }
        let newLen = max(seqLen, 4096)
        let (c, s) = TadaRoPE.buildRopeCache(
            seqLen: newLen, headDim: config.headDim,
            ropeTheta: config.ropeTheta, ropeScaling: config.ropeScaling
        )
        ropeCos = c
        ropeSin = s
        ropeLen = newLen
    }

    func getRopeSlice(start: Int, length: Int) -> (MLXArray, MLXArray) {
        ensureRope(seqLen: start + length)
        return (ropeCos![start..<(start + length)], ropeSin![start..<(start + length)])
    }

    func lmHeadForward(_ h: MLXArray) -> MLXArray {
        if config.tieWordEmbeddings {
            return h.matmul(model.embedTokens.weight.T)
        }
        return lmHead!(h)
    }

    func applyBottleneck(_ h: MLXArray) -> MLXArray {
        if let proj = bottleneckProj { return proj(h) }
        return h
    }

    // MARK: - Forward One Step

    func forwardOneStep(
        inputIds: MLXArray,
        acousticFeatures: MLXArray,
        acousticMasks: MLXArray,
        timeLenBefore: MLXArray,
        timeLenAfter: MLXArray,
        cache: [TadaKVCache]?
    ) -> (hidden: MLXArray, logits: MLXArray) {
        let inputsEmbeds = model.embedTokens(inputIds)
            + acousticProj(acousticFeatures)
            + acousticMaskEmb(acousticMasks)
            + timeStartEmbed(timeLenBefore)
            + timeEndEmbed(timeLenAfter)

        let cacheLen = cache?.first?.seqLen ?? 0
        let seqLen = inputsEmbeds.shape[1]
        let (cos, sin) = getRopeSlice(start: cacheLen, length: seqLen)

        let mask: MLXArray?
        if seqLen > 1 {
            let total = cacheLen + seqLen
            let rowIdx = MLXArray(0..<seqLen).reshaped(-1, 1) + cacheLen
            let colIdx = MLXArray(0..<total).reshaped(1, -1)
            let m = which(colIdx .<= rowIdx, MLXArray(0.0), MLXArray(-1e9))
            mask = expandedDimensions(expandedDimensions(m, axis: 0), axis: 0)
        } else {
            mask = nil
        }

        let hidden = model(inputsEmbeds, cos: cos, sin: sin, mask: mask, cache: cache)
        let logits = lmHeadForward(hidden)
        return (hidden, logits)
    }

    // MARK: - Build Prompt Embeds

    func buildPromptInputsEmbeds(
        inputIds: MLXArray,
        paf: MLXArray, pam: MLXArray,
        tlb: MLXArray, tla: MLXArray,
        promptLen: Int
    ) -> MLXArray {
        let B = inputIds.shape[0]
        let shift = config.shiftAcoustic
        let tokenEmb = model.embedTokens(inputIds[0..., ..<promptLen])

        let nAc = min(promptLen - shift - 1, paf.shape[1])
        var acousticFull: MLXArray
        var masksFull: MLXArray

        if nAc > 0 {
            var afParts = [MLXArray]()
            afParts.append(MLXArray.zeros([B, shift + 1, config.acousticDim]))
            afParts.append(paf[0..., ..<nAc])
            let remaining = promptLen - shift - 1 - nAc
            if remaining > 0 {
                afParts.append(MLXArray.zeros([B, remaining, config.acousticDim]))
            }
            acousticFull = concatenated(afParts, axis: 1)

            var mParts = [MLXArray]()
            mParts.append(MLXArray.zeros([B, shift + 1], dtype: .int32))
            mParts.append(pam[0..., ..<nAc])
            if remaining > 0 {
                mParts.append(MLXArray.zeros([B, remaining], dtype: .int32))
            }
            masksFull = concatenated(mParts, axis: 1)
        } else {
            acousticFull = MLXArray.zeros([B, promptLen, config.acousticDim])
            masksFull = MLXArray.zeros([B, promptLen], dtype: .int32)
        }

        let acousticEmb = acousticProj(acousticFull) + acousticMaskEmb(masksFull)

        let nT = min(promptLen - shift - 1, tlb.shape[1] - 1)
        var timeBefore: MLXArray
        var timeAfter: MLXArray

        if nT > 0 {
            var tbParts = [MLXArray]()
            tbParts.append(MLXArray.zeros([B, shift + 1], dtype: .int32))
            tbParts.append(tlb[0..., 1..<(1 + nT)])
            let remaining = promptLen - shift - 1 - nT
            if remaining > 0 {
                tbParts.append(MLXArray.zeros([B, remaining], dtype: .int32))
            }
            timeBefore = concatenated(tbParts, axis: 1)

            var taParts = [MLXArray]()
            taParts.append(MLXArray.zeros([B, shift + 1], dtype: .int32))
            taParts.append(tla[0..., 1..<(1 + nT)])
            if remaining > 0 {
                taParts.append(MLXArray.zeros([B, remaining], dtype: .int32))
            }
            timeAfter = concatenated(taParts, axis: 1)
        } else {
            timeBefore = MLXArray.zeros([B, promptLen], dtype: .int32)
            timeAfter = MLXArray.zeros([B, promptLen], dtype: .int32)
        }

        let timeEmb = timeStartEmbed(timeBefore) + timeEndEmbed(timeAfter)
        return tokenEmb + acousticEmb + timeEmb
    }

    // MARK: - Token Sampling

    static func sampleToken(
        logits: MLXArray,
        inputIds: MLXArray,
        opts: TadaInferenceOptions,
        padTokenId: Int
    ) -> MLXArray {
        var logitsM = logits
        logitsM = logitsM.at[0..., padTokenId].add(MLXArray(-1e9))

        if opts.textDoSample {
            logitsM = logitsM / opts.textTemperature

            if opts.textTopK > 0 {
                let topK = min(opts.textTopK, logitsM.shape[logitsM.ndim - 1])
                let sorted = MLX.sorted(logitsM, axis: -1)
                let kth = sorted[.ellipsis, (-topK)..<(-topK + 1)]
                logitsM = which(logitsM .< kth, MLXArray(-1e9), logitsM)
            }

            let probs = softmax(logitsM, axis: -1)
            let nextToken = MLXRandom.categorical(log(probs + 1e-12))
            return expandedDimensions(nextToken, axis: -1)
        } else {
            return expandedDimensions(argMax(logitsM, axis: -1), axis: -1)
        }
    }

    // MARK: - Build Inputs

    func buildInputs(
        text: String,
        reference: TadaReference,
        numTransitionSteps: Int,
        tokenizer: Tokenizers.Tokenizer
    ) -> (inputIds: MLXArray, paf: MLXArray, pam: MLXArray, tlb: MLXArray, tla: MLXArray, prefillLen: Int) {
        let shift = config.shiftAcoustic
        let bosId = config.bosTokenId
        let eotId = config.eotId

        let refTokenIds = tokenizer.encode(text: reference.text)
        let targetTokenIds = tokenizer.encode(text: " " + text)

        let tokenPositionsMx = reference.tokenPositions
        let posPadded = concatenated([MLXArray.ones([1, 1], dtype: tokenPositionsMx.dtype), tokenPositionsMx], axis: 1)
        var timeGaps = clip(tokenPositionsMx - posPadded[0..., ..<(-1)], min: 0, max: config.numTimeClasses - 1)
        timeGaps = concatenated([MLXArray.zeros([1, 1], dtype: timeGaps.dtype), timeGaps], axis: 1)
        var tlb = timeGaps[0..., ..<(-1)].asType(.int32)
        var tla = timeGaps[0..., 1...].asType(.int32)
        var paf = reference.tokenValues
        var pam = MLXArray.ones([paf.shape[0], paf.shape[1]], dtype: .int32)

        let prefixText = "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        let prefixIds = tokenizer.encode(text: prefixText)
        let prefixLen = prefixIds.count

        let prefillTokenIds = [bosId] + prefixIds + refTokenIds

        paf = padded(paf, widths: [.init((0, 0)), .init((prefixLen, 0)), .init((0, 0))])
        pam = padded(pam, widths: [.init((0, 0)), .init((prefixLen, 0))])
        tlb = padded(tlb, widths: [.init((0, 0)), .init((prefixLen, 0))])
        tla = padded(tla, widths: [.init((0, 0)), .init((prefixLen, 0))])

        if numTransitionSteps > 0 {
            paf = paf[0..., ..<(-numTransitionSteps), 0...]
            pam = pam[0..., ..<(-numTransitionSteps)]
            tlb = tlb[0..., ..<(-numTransitionSteps)]
            tla = tla[0..., ..<(-numTransitionSteps)]
        }

        pam = concatenated([pam[0..., 1...], MLXArray.ones([pam.shape[0], 1], dtype: pam.dtype)], axis: 1)

        let fullIds = prefillTokenIds + targetTokenIds + Array(repeating: eotId, count: shift)
        var inputIds = MLXArray(fullIds.map { Int32($0) }, [1, fullIds.count])

        let promptTokenLen = paf.shape[1]
        let promptIds = inputIds[0..., ..<promptTokenLen]

        let isStart = promptIds .== config.startHeaderId
        let isEnd = promptIds .== config.endHeaderId
        let headerDepth = cumsum(isStart.asType(DType.int32), axis: 1) - cumsum(isEnd.asType(DType.int32), axis: 1)
        let inHeader = (headerDepth .> 0) | isStart | isEnd
        let isStructural = inHeader | (promptIds .== eotId) | (promptIds .== bosId) | (promptIds .== 128001)
        let maskedIds = which(isStructural, promptIds, MLXArray.full(promptIds.shape, values: MLXArray(Int32(config.padTokenId))))
        inputIds = concatenated([maskedIds, inputIds[0..., promptTokenLen...]], axis: 1)

        let nAc = min(prefillTokenIds.count - shift - 1, paf.shape[1])
        let nT = min(prefillTokenIds.count - shift - 1, tlb.shape[1] - 1)
        let nFramesCap = max(0, tlb.shape[1] - 2)
        let nPrefill = (nAc > 0 && nT > 0) ? min(nAc, nT, nFramesCap) : 0
        let computedPrefillLen = nPrefill > 0 ? min(prefillTokenIds.count, shift + nPrefill + 1) : 0

        return (inputIds, paf, pam, tlb, tla, computedPrefillLen)
    }

    // MARK: - Prefill

    func prefill(
        inputIds: MLXArray,
        paf: MLXArray, pam: MLXArray,
        tlb: MLXArray, tla: MLXArray,
        prefillLen: Int
    ) -> [TadaKVCache] {
        let cache = (0..<config.numHiddenLayers).map { _ in TadaKVCache() }
        guard prefillLen > 0 else { return cache }

        let embeds = buildPromptInputsEmbeds(
            inputIds: inputIds, paf: paf, pam: pam, tlb: tlb, tla: tla, promptLen: prefillLen
        )
        let dualEmbeds = concatenated([embeds, embeds], axis: 0)
        let (cos, sin) = getRopeSlice(start: 0, length: prefillLen)

        let rowIdx = MLXArray(0..<prefillLen).reshaped(-1, 1)
        let colIdx = MLXArray(0..<prefillLen).reshaped(1, -1)
        let cmask = which(colIdx .<= rowIdx, MLXArray(0.0), MLXArray(-1e9))
        let maskExpanded = expandedDimensions(expandedDimensions(cmask, axis: 0), axis: 0)

        let hidden = model(dualEmbeds, cos: cos, sin: sin, mask: maskExpanded, cache: cache)
        eval(hidden)
        for layerCache in cache {
            if let k = layerCache.keys, let v = layerCache.values {
                eval(k, v)
            }
        }

        return cache
    }

    // MARK: - Autoregressive Loop

    func autoregressiveLoop(
        inputIds: inout MLXArray,
        paf: MLXArray, pam: MLXArray,
        tlb: MLXArray, tla: MLXArray,
        prefillLen: Int,
        cache: [TadaKVCache],
        opts: TadaInferenceOptions,
        numExtraSteps: Int = 0
    ) -> (acoustic: [MLXArray], timeBefore: [MLXArray], tokenIds: [MLXArray]) {
        let shift = config.shiftAcoustic
        let padId = config.padTokenId
        let B = 1
        let nPf = prefillLen - shift

        var acousticFeaturesVal: MLXArray
        var acousticMasksVal: MLXArray
        var timeBeforeVal: MLXArray
        var timeAfterVal: MLXArray

        if nPf > 0 {
            acousticFeaturesVal = expandedDimensions(paf[0..., nPf - 1], axis: 1)
            acousticMasksVal = expandedDimensions(pam[0..., nPf - 1], axis: 1)
        } else {
            acousticFeaturesVal = MLXArray.zeros([B, 1, config.acousticDim])
            acousticMasksVal = MLXArray.zeros([B, 1], dtype: .int32)
        }

        if nPf > 0 && nPf < tlb.shape[1] {
            timeBeforeVal = expandedDimensions(tlb[0..., nPf], axis: 1)
            timeAfterVal = expandedDimensions(tla[0..., nPf], axis: 1)
        } else {
            timeBeforeVal = MLXArray.zeros([B, 1], dtype: .int32)
            timeAfterVal = MLXArray.zeros([B, 1], dtype: .int32)
        }

        var negCond = MLXArray.zeros([B, config.hiddenSize])
        var allAcoustic = [MLXArray]()
        var allTimeBefore = [MLXArray]()
        var allTokenIds = [MLXArray]()
        var lastTimeBefore: MLXArray?

        for i in 0..<nPf {
            allAcoustic.append(expandedDimensions(paf[0..., i], axis: 1))
        }
        for i in 0..<nPf {
            allTimeBefore.append(expandedDimensions(tlb[0..., i + 1], axis: 1))
        }

        let maxSteps = inputIds.shape[1] + numExtraSteps
        for step in prefillLen..<maxSteps {
            let inputSlice = inputIds[0..., step..<(step + 1)]
            let isStructural = (inputSlice .== config.startHeaderId)
                | (inputSlice .== config.endHeaderId)
                | (inputSlice .== config.eotId)
            let negSlice = which(isStructural, inputSlice, MLXArray.full(inputSlice.shape, values: MLXArray(Int32(padId))))

            let combinedSlice = concatenated([inputSlice, negSlice], axis: 0)
            let combinedAcoustic = concatenated([acousticFeaturesVal, acousticFeaturesVal], axis: 0)
            let combinedMasks = concatenated([acousticMasksVal, acousticMasksVal], axis: 0)
            let combinedTb = concatenated([timeBeforeVal, timeBeforeVal], axis: 0)
            let combinedTa = concatenated([timeAfterVal, timeAfterVal], axis: 0)

            let (hidden, logits) = forwardOneStep(
                inputIds: combinedSlice,
                acousticFeatures: combinedAcoustic,
                acousticMasks: combinedMasks,
                timeLenBefore: combinedTb,
                timeLenAfter: combinedTa,
                cache: cache
            )

            negCond = hidden[B..<(2 * B)]
            let hiddenPos = hidden[..<B]
            let logitsPos = logits[..<B]

            let noise = MLXRandom.normal([B, config.acousticDim + timeDim]) * opts.noiseTemperature
            let speech = predictionHead.solve(
                noise: noise,
                cond: hiddenPos,
                negCond: negCond,
                acousticDim: config.acousticDim,
                numSteps: opts.numFlowMatchingSteps,
                acousticCfgScale: opts.acousticCfgScale,
                durationCfgScale: opts.durationCfgScale,
                cfgSchedule: opts.cfgSchedule,
                timeSchedule: opts.timeSchedule,
                bottleneckFn: { [self] h in applyBottleneck(h) }
            )

            let timeGray = speech[.ellipsis, (-timeDim)...]
            let predTb = expandedDimensions(
                TadaGrayCode.decodeGrayCodeToTime(timeGray[.ellipsis, ..<numTimeBits], numBits: numTimeBits), axis: 0
            )
            let predTa = expandedDimensions(
                TadaGrayCode.decodeGrayCodeToTime(timeGray[.ellipsis, numTimeBits...], numBits: numTimeBits), axis: 0
            )

            if step >= inputIds.shape[1] - 1 {
                let nextToken = Self.sampleToken(logits: logitsPos[0..., -1, 0...], inputIds: inputIds, opts: opts, padTokenId: padId)
                inputIds = concatenated([inputIds, nextToken.asType(.int32)], axis: 1)
                allTokenIds.append(nextToken)
                if config.eosTokenId.contains(nextToken[0, 0].item(Int.self)) {
                    break
                }
            } else {
                allTokenIds.append(inputIds[0..., (step + 1)..<(step + 2)])
            }

            if step >= shift {
                if step - shift < paf.shape[1] {
                    acousticFeaturesVal = expandedDimensions(paf[0..., step - shift], axis: 1)
                    acousticMasksVal = expandedDimensions(pam[0..., step - shift], axis: 1)
                } else {
                    acousticFeaturesVal = expandedDimensions(speech[.ellipsis, ..<config.acousticDim], axis: 0)
                    acousticMasksVal = MLXArray.ones([B, 1], dtype: .int32)
                }
                allAcoustic.append(acousticFeaturesVal)

                if step - shift < tlb.shape[1] - 1 {
                    timeBeforeVal = expandedDimensions(tlb[0..., step - shift + 1], axis: 1)
                    timeAfterVal = expandedDimensions(tla[0..., step - shift + 1], axis: 1)
                } else {
                    timeBeforeVal = predTb
                    timeAfterVal = predTa
                }
                allTimeBefore.append(timeBeforeVal)
                lastTimeBefore = timeBeforeVal
            }

            eval(inputIds, acousticFeaturesVal, timeBeforeVal)
        }

        if let lt = lastTimeBefore {
            allTimeBefore.append(lt)
        }

        return (allAcoustic, allTimeBefore, allTokenIds)
    }

    // MARK: - Decode Output

    func decodeOutput(
        allAcoustic: [MLXArray],
        allTimeBefore: [MLXArray],
        numPromptTokens: Int,
        numTransitionSteps: Int
    ) -> MLXArray {
        let outAcoustic: MLXArray
        if !allAcoustic.isEmpty {
            outAcoustic = concatenated(
                allAcoustic.map { $0.ndim == 3 ? $0 : expandedDimensions($0, axis: 1) },
                axis: 1
            )
        } else {
            outAcoustic = MLXArray.zeros([1, 0, config.acousticDim])
        }

        let outTime: MLXArray
        if !allTimeBefore.isEmpty {
            outTime = concatenated(
                allTimeBefore.map { $0.ndim == 2 ? $0 : expandedDimensions($0, axis: 1) },
                axis: 1
            )
        } else {
            outTime = MLXArray.zeros([1, 0], dtype: .int32)
        }

        let acousticFeatures = outAcoustic * config.acousticStd + config.acousticMean
        let startIdx = numPromptTokens + numTransitionSteps - 1
        let encoded = acousticFeatures[0..., startIdx..., 0...]
        let timeBeforeOut = outTime[0..., startIdx...]

        guard let decoder else {
            fatalError("Decoder not loaded")
        }

        let wav = decoder.decodeFrames(encoded: encoded[0], timeBefore: timeBeforeOut[0])
        let leadFrames = timeBeforeOut[0, 0].item(Int.self)
        let leadSamples = Int(24000 * leadFrames / 50)

        if leadSamples > 0 && leadSamples < wav.shape[0] {
            return wav[leadSamples...]
        }
        return wav
    }

    // MARK: - Generate

    func generate(
        text: String,
        reference: TadaReference,
        inferenceOptions: TadaInferenceOptions? = nil,
        numTransitionSteps: Int = 5,
        numExtraSteps: Int = 0
    ) -> (audio: MLXArray, numTokens: Int) {
        let opts = inferenceOptions ?? TadaInferenceOptions()
        let normalizedText = TadaTextNormalizer.normalize(text)

        guard let tokenizer else {
            fatalError("Tokenizer not loaded")
        }

        var (inputIds, paf, pam, tlb, tla, prefillLen) = buildInputs(
            text: normalizedText, reference: reference,
            numTransitionSteps: numTransitionSteps, tokenizer: tokenizer
        )

        let cache = prefill(inputIds: inputIds, paf: paf, pam: pam, tlb: tlb, tla: tla, prefillLen: prefillLen)

        let (allAcoustic, allTimeBefore, allTokenIds) = autoregressiveLoop(
            inputIds: &inputIds, paf: paf, pam: pam, tlb: tlb, tla: tla,
            prefillLen: prefillLen, cache: cache, opts: opts, numExtraSteps: numExtraSteps
        )

        let wav = decodeOutput(
            allAcoustic: allAcoustic,
            allTimeBefore: allTimeBefore,
            numPromptTokens: paf.shape[1],
            numTransitionSteps: numTransitionSteps
        )

        return (wav, allTokenIds.count)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if config.tieWordEmbeddings {
            w["lm_head.weight"] = nil
        }
        return w
    }
}

// MARK: - TadaTTSModel (SpeechGenerationModel conformance)

public class TadaTTSModel: @unchecked Sendable, SpeechGenerationModel {
    private let inner: TadaForCausalLM
    public var inferenceOptions = TadaInferenceOptions()

    public var sampleRate: Int { 24000 }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters()
    }

    init(inner: TadaForCausalLM) {
        self.inner = inner
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
        guard let refAudio, let refText else {
            throw AudioGenerationError.generationFailed("TADA requires reference audio and text for voice cloning")
        }

        let reference = try await encodeReference(audio: refAudio, text: refText)
        let (wav, _) = inner.generate(text: text, reference: reference, inferenceOptions: inferenceOptions)
        return wav
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
        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.generationFailed("Model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: refAudio,
                    refText: refText, language: language,
                    generationParameters: generationParameters
                )
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - Reference Encoding

    func encodeReference(audio: MLXArray, text: String) async throws -> TadaReference {
        guard let encoder = inner.encoder else {
            throw AudioGenerationError.generationFailed("TADA encoder not loaded")
        }
        guard let aligner = inner.aligner else {
            throw AudioGenerationError.generationFailed("TADA aligner not loaded")
        }
        guard let tokenizer = inner.tokenizer else {
            throw AudioGenerationError.generationFailed("TADA tokenizer not loaded")
        }

        let audio24k = audio.ndim == 1 ? expandedDimensions(audio, axis: 0) : audio

        let samples24k = audio24k[0].asArray(Float.self)
        let samples16k = try resampleAudio(samples24k, from: 24000, to: 16000)
        let audio16k = MLXArray(samples16k, [1, samples16k.count])

        let textTokenIds = tokenizer.encode(text: text)
        let eosTokenId = inner.config.eosTokenId.first ?? 128001

        let (positions, masks) = aligner.align(
            audio16k: audio16k, textTokens: [textTokenIds],
            inputLengths: [audio24k.shape[1]], eosTokenId: eosTokenId
        )

        let positionsArr = MLXArray(positions[0].map { Int32($0) }, [1, positions[0].count])
        let masksArr = MLXArray(masks[0].map { Int32($0) }, [1, masks[0].count])
        let textTokensArr = MLXArray(textTokenIds.map { Int32($0) }, [1, textTokenIds.count])

        let encoderOutput = encoder.encode(
            audio: audio24k,
            tokenPositions: positionsArr,
            tokenMasks: masksArr,
            text: [text],
            textTokens: textTokensArr,
            textTokensLen: MLXArray([Int32(textTokenIds.count)])
        )

        return TadaReference(
            tokenValues: encoderOutput.tokenValues,
            tokenPositions: encoderOutput.tokenPositions,
            tokenMasks: encoderOutput.tokenMasks,
            textTokens: textTokensArr,
            textTokensLen: MLXArray([Int32(textTokenIds.count)]),
            audioLen: encoderOutput.audioLen,
            text: text
        )
    }

    // MARK: - fromPretrained

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> TadaTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.generationFailed("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors", cache: cache
        )

        let configPath = modelDir.appendingPathComponent("model").appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(TadaConfig.self, from: configData)

        let inner = TadaForCausalLM(config: config)

        // Load model weights — @ModuleInfo(key:) handles snake_case→camelCase mapping
        let modelWeights = try loadSafetensors(modelDir.appendingPathComponent("model"))
        try inner.update(parameters: ModuleParameters.unflattened(inner.sanitize(weights: modelWeights)), verify: .noUnusedKeys)
        eval(inner)

        // Load decoder weights
        let decoder = TadaDecoder()
        let decoderWeights = try loadSafetensors(modelDir.appendingPathComponent("decoder"))
        try decoder.update(parameters: ModuleParameters.unflattened(decoderWeights), verify: .noUnusedKeys)
        eval(decoder)
        inner.decoder = decoder

        // Load encoder weights
        let encoder = TadaEncoder()
        let encoderWeights = try loadSafetensors(modelDir.appendingPathComponent("encoder"))
        try encoder.update(parameters: ModuleParameters.unflattened(encoderWeights), verify: .noUnusedKeys)
        eval(encoder)
        inner.encoder = encoder

        // Load aligner weights
        let aligner = TadaAligner()
        let alignerWeights = try loadSafetensors(modelDir.appendingPathComponent("aligner"))
        try aligner.update(parameters: ModuleParameters.unflattened(alignerWeights), verify: .noUnusedKeys)
        eval(aligner)
        inner.aligner = aligner

        // Load tokenizer — TADA doesn't bundle one; it uses the base Llama 3.2 tokenizer.
        let tokenizerDir = modelDir.appendingPathComponent("model")
        if FileManager.default.fileExists(atPath: tokenizerDir.appendingPathComponent("tokenizer.json").path)
            || FileManager.default.fileExists(atPath: tokenizerDir.appendingPathComponent("tokenizer.model").path)
        {
            inner.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        } else {
            inner.tokenizer = try await AutoTokenizer.from(pretrained: "NousResearch/Llama-3.2-1B")
        }

        return TadaTTSModel(inner: inner)
    }
}

// MARK: - Weight Loading Helper

private func loadSafetensors(_ dir: URL) throws -> [String: MLXArray] {
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}
