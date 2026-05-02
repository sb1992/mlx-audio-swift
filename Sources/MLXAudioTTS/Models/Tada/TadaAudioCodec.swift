import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Constants

private let kHiddenDim = 1024
private let kEmbedDim = 512
private let kEncoderStrides = [6, 5, 4, 4]
private let kDecoderStrides = [4, 4, 5, 6]
private let kNumAttnLayers = 6
private let kNumAttnHeads = 8
private let kAttnDimFeedforward = 4096
private let kSampleStd: Float = 0.5
private let kAcousticMean: Float = 0.0
private let kAcousticStd: Float = 1.5
private let kWavDecoderChannels = 1536

// MARK: - Snake1d Activation

class TadaSnake1d: Module {
    var alpha: MLXArray

    init(channels: Int) {
        self.alpha = MLXArray.ones([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1.0 / alpha) * pow(sin(alpha * x), 2)
    }
}

// MARK: - Residual Unit

class TadaResidualUnit: Module {
    var snake1: TadaSnake1d
    var conv1: Conv1d
    var snake2: TadaSnake1d
    var conv2: Conv1d

    init(dim: Int = 16, dilation: Int = 1) {
        let pad = ((7 - 1) * dilation) / 2
        self.snake1 = TadaSnake1d(channels: dim)
        self.conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: pad, dilation: dilation)
        self.snake2 = TadaSnake1d(channels: dim)
        self.conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv2(snake2(conv1(snake1(x))))
        let diff = x.shape[1] - y.shape[1]
        if diff > 0 {
            let half = diff / 2
            return x[0..., half..<(half + y.shape[1]), 0...] + y
        }
        return x + y
    }
}

// MARK: - Segment Attention Masks

func tadaCreateSegmentAttentionMask(_ textTokenMask: MLXArray) -> MLXArray {
    let blockIds = cumsum(textTokenMask, axis: 1)
    let blockI = expandedDimensions(blockIds, axis: 2)
    let blockJ = expandedDimensions(blockIds, axis: 1)
    let sameBlock = blockI .== blockJ
    let isMarkedI = expandedDimensions(textTokenMask, axis: 2).asType(.bool)
    let isMarkedJ = expandedDimensions(textTokenMask, axis: 1).asType(.bool)
    let sameValid = sameBlock & (.!isMarkedJ | (isMarkedI & sameBlock))
    let prevBlock = blockJ .== (blockI - 1)
    let prevValid = prevBlock & .!isMarkedJ
    let canAttend = sameValid | (isMarkedI & prevValid)
    return .!canAttend
}

private func decoderCreateSegmentAttentionMask(_ textTokenMask: MLXArray) -> MLXArray {
    let blockIds = cumsum(textTokenMask, axis: 1) - textTokenMask
    let blockI = expandedDimensions(blockIds, axis: 2)
    let blockJ = expandedDimensions(blockIds, axis: 1)
    let sameBlock = blockI .== blockJ
    let prevBlock = blockJ .== (blockI - 1)
    let canAttend = sameBlock | prevBlock
    return .!canAttend
}

// MARK: - Local Self-Attention (with RoPE)

class TadaLocalSelfAttention: Module {
    let dModel: Int
    let numHeads: Int
    let headDim: Int
    var qkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    let ropeCos: MLXArray
    let ropeSin: MLXArray

    init(dModel: Int, numHeads: Int = 8, maxSeqLen: Int = 8192) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.headDim = dModel / numHeads
        self.qkv = Linear(dModel, 3 * dModel)
        self._outProj.wrappedValue = Linear(dModel, dModel)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: dModel)

        let invFreq = 1.0 / pow(
            MLXArray(10000.0),
            MLXArray(stride(from: 0, to: dModel / numHeads, by: 2).map { Float($0) }) / Float(dModel / numHeads)
        )
        let positions = MLXArray(0..<maxSeqLen).asType(.float32)
        let freqs = outer(positions, invFreq)
        self.ropeCos = cos(freqs)
        self.ropeSin = sin(freqs)
    }

    private func applyRope(_ x: MLXArray, seqLen: Int) -> MLXArray {
        let (B, H, S, D) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let cosSlice = expandedDimensions(expandedDimensions(ropeCos[..<seqLen], axis: 0), axis: 0)
        let sinSlice = expandedDimensions(expandedDimensions(ropeSin[..<seqLen], axis: 0), axis: 0)
        let xPairs = x.reshaped(B, H, S, D / 2, 2)
        let x0 = xPairs[.ellipsis, 0]
        let x1 = xPairs[.ellipsis, 1]
        let r0 = x0 * cosSlice - x1 * sinSlice
        let r1 = x0 * sinSlice + x1 * cosSlice
        return stacked([r0, r1], axis: -1).reshaped(B, H, S, D)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, S, D) = (x.shape[0], x.shape[1], x.shape[2])
        let qkvOut = qkv(x).reshaped(B, S, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]
        q = applyRope(q, seqLen: S)
        k = applyRope(k, seqLen: S)
        let scale = pow(Float(headDim), -0.5)
        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale

        if let mask {
            if mask.ndim == 2 {
                let maskExpanded = expandedDimensions(expandedDimensions(
                    which(mask, MLXArray(-1e9), MLXArray(0.0)), axis: 0), axis: 0)
                attn = attn + maskExpanded
            } else if mask.ndim == 3 {
                let maskExpanded = expandedDimensions(
                    which(mask, MLXArray(-1e9), MLXArray(0.0)), axis: 1)
                attn = attn + maskExpanded
            }
        }

        attn = softmax(attn, axis: -1)
        let out = attn.matmul(v).transposed(0, 2, 1, 3).reshaped(B, S, D)
        return layerNorm(x + outProj(out))
    }
}

// MARK: - Local Attention Encoder Layer

class TadaLocalAttentionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TadaLocalSelfAttention
    var linear1: Linear
    var linear2: Linear
    var norm: LayerNorm

    init(dModel: Int, numHeads: Int = 8, dFf: Int = 4096) {
        self._selfAttn.wrappedValue = TadaLocalSelfAttention(dModel: dModel, numHeads: numHeads)
        self.linear1 = Linear(dModel, dFf)
        self.linear2 = Linear(dFf, dModel)
        self.norm = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attended = selfAttn(x, mask: mask)
        let ffnOut = linear2(gelu(linear1(attended)))
        return norm(attended + ffnOut)
    }
}

// MARK: - Local Attention Encoder

class TadaLocalAttentionEncoder: Module {
    var layers: [TadaLocalAttentionEncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(dModel: Int, numLayers: Int = 4, numHeads: Int = 8, dFf: Int = 4096) {
        self.layers = (0..<numLayers).map { _ in
            TadaLocalAttentionEncoderLayer(dModel: dModel, numHeads: numHeads, dFf: dFf)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x
        for layer in layers { out = layer(out, mask: mask) }
        return finalNorm(out)
    }
}

// MARK: - Encoder Block

class TadaEncoderBlock: Module {
    var res1: TadaResidualUnit
    var res2: TadaResidualUnit
    var res3: TadaResidualUnit
    var snake: TadaSnake1d
    var conv: Conv1d

    init(dim: Int = 16, stride: Int = 1) {
        self.res1 = TadaResidualUnit(dim: dim / 2, dilation: 1)
        self.res2 = TadaResidualUnit(dim: dim / 2, dilation: 3)
        self.res3 = TadaResidualUnit(dim: dim / 2, dilation: 9)
        self.snake = TadaSnake1d(channels: dim / 2)
        let padding = Int(ceil(Double(stride) / 2.0))
        self.conv = Conv1d(inputChannels: dim / 2, outputChannels: dim, kernelSize: 2 * stride, stride: stride, padding: padding)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = res1(x)
        out = res2(out)
        out = res3(out)
        out = snake(out)
        return conv(out)
    }
}

// MARK: - WavEncoder

class TadaWavEncoder: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: Conv1d
    var blocks: [TadaEncoderBlock]
    @ModuleInfo(key: "final_snake") var finalSnake: TadaSnake1d
    @ModuleInfo(key: "final_conv") var finalConv: Conv1d

    override init() {
        var dModel = 64
        self._initialConv.wrappedValue = Conv1d(inputChannels: 1, outputChannels: dModel, kernelSize: 7, padding: 3)
        var blocks = [TadaEncoderBlock]()
        for stride in kEncoderStrides {
            dModel *= 2
            blocks.append(TadaEncoderBlock(dim: dModel, stride: stride))
        }
        self.blocks = blocks
        self._finalSnake.wrappedValue = TadaSnake1d(channels: dModel)
        self._finalConv.wrappedValue = Conv1d(inputChannels: dModel, outputChannels: kHiddenDim, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = initialConv(x)
        for blk in blocks { out = blk(out) }
        out = finalSnake(out)
        return finalConv(out)
    }
}

// MARK: - Encoder Output

struct TadaEncoderOutput {
    let audioLen: MLXArray
    let text: [String]
    let tokenPositions: MLXArray
    let tokenValues: MLXArray
    let sampleRate: Int
    let textTokens: MLXArray?
    let textTokensLen: MLXArray?
    let tokenMasks: MLXArray?
}

// MARK: - Encoder (top-level)

class TadaEncoder: Module {
    @ModuleInfo(key: "wav_encoder") var wavEncoder: TadaWavEncoder
    @ModuleInfo(key: "local_attention_encoder") var localAttentionEncoder: TadaLocalAttentionEncoder
    @ModuleInfo(key: "hidden_linear") var hiddenLinear: Linear
    @ModuleInfo(key: "pos_emb") var posEmb: Embedding

    override init() {
        self._wavEncoder.wrappedValue = TadaWavEncoder()
        self._localAttentionEncoder.wrappedValue = TadaLocalAttentionEncoder(
            dModel: kHiddenDim, numLayers: kNumAttnLayers, numHeads: kNumAttnHeads, dFf: kAttnDimFeedforward
        )
        self._hiddenLinear.wrappedValue = Linear(kHiddenDim, kEmbedDim)
        self._posEmb.wrappedValue = Embedding(embeddingCount: 2, dimensions: kHiddenDim)
    }

    func getEncoderOutputs(audio: MLXArray, tokenMasks: MLXArray) -> (MLXArray, MLXArray) {
        var x = expandedDimensions(audio, axis: -1)
        x = padded(x, widths: [.init((0, 0)), .init((0, 960)), .init((0, 0))])
        var encOut = wavEncoder(x)
        let seqLen = encOut.shape[1]
        let padLen = seqLen - tokenMasks.shape[1]

        let paddedMasks: MLXArray
        if padLen > 0 {
            paddedMasks = padded(tokenMasks, widths: [.init((0, 0)), .init((0, padLen))])
        } else {
            paddedMasks = tokenMasks[0..., ..<seqLen]
        }

        encOut = encOut + posEmb(paddedMasks.asType(.int32))
        let attnMask = tadaCreateSegmentAttentionMask(paddedMasks)
        encOut = localAttentionEncoder(encOut, mask: attnMask)
        encOut = hiddenLinear(encOut)
        return (encOut, paddedMasks)
    }

    func encode(
        audio: MLXArray,
        tokenPositions: MLXArray,
        tokenMasks: MLXArray,
        audioLength: MLXArray? = nil,
        text: [String]? = nil,
        textTokens: MLXArray? = nil,
        textTokensLen: MLXArray? = nil,
        sample: Bool = true
    ) -> TadaEncoderOutput {
        let audioLen = audioLength ?? MLXArray([audio.shape[1]])
        let (encOut, paddedMasks) = getEncoderOutputs(audio: audio, tokenMasks: tokenMasks)

        var encodedExpanded = which(
            expandedDimensions(paddedMasks, axis: -1) .== 0,
            MLXArray.zeros(like: encOut),
            encOut
        )

        if kSampleStd > 0.0 && sample {
            let noise = MLXRandom.normal(encodedExpanded.shape) * kSampleStd
            encodedExpanded = which(
                expandedDimensions(paddedMasks, axis: -1) .== 0,
                encodedExpanded,
                encodedExpanded + noise
            )
        }

        let positionsClamped = clip(tokenPositions - 1, min: 0, max: encodedExpanded.shape[1] - 1)
        let (B, T) = (positionsClamped.shape[0], positionsClamped.shape[1])
        let batchIdx = repeated(MLXArray(0..<B).reshaped(B, 1), count: T, axis: 1)
        let tokenValues = encodedExpanded[batchIdx, positionsClamped]
        let normalizedValues = (tokenValues - kAcousticMean) / kAcousticStd

        return TadaEncoderOutput(
            audioLen: audioLen,
            text: text ?? [""],
            tokenPositions: tokenPositions,
            tokenValues: normalizedValues,
            sampleRate: 24000,
            textTokens: textTokens,
            textTokensLen: textTokensLen,
            tokenMasks: paddedMasks
        )
    }
}

// MARK: - Decoder Block

class TadaDecoderBlock: Module {
    var snake: TadaSnake1d
    @ModuleInfo(key: "conv_transpose") var convTranspose: ConvTransposed1d
    var res1: TadaResidualUnit
    var res2: TadaResidualUnit
    var res3: TadaResidualUnit

    init(inputDim: Int = 16, outputDim: Int = 8, stride: Int = 1) {
        self.snake = TadaSnake1d(channels: inputDim)
        let padding = Int(ceil(Double(stride) / 2.0))
        self._convTranspose.wrappedValue = ConvTransposed1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: 2 * stride, stride: stride, padding: padding
        )
        self.res1 = TadaResidualUnit(dim: outputDim, dilation: 1)
        self.res2 = TadaResidualUnit(dim: outputDim, dilation: 3)
        self.res3 = TadaResidualUnit(dim: outputDim, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = snake(x)
        out = convTranspose(out)
        out = res1(out)
        out = res2(out)
        out = res3(out)
        return out
    }
}

// MARK: - DAC Decoder

class TadaDACDecoder: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: Conv1d
    var blocks: [TadaDecoderBlock]
    @ModuleInfo(key: "final_snake") var finalSnake: TadaSnake1d
    @ModuleInfo(key: "final_conv") var finalConv: Conv1d

    init(inputChannel: Int, channels: Int, rates: [Int] = kDecoderStrides, dOut: Int = 1) {
        self._initialConv.wrappedValue = Conv1d(inputChannels: inputChannel, outputChannels: channels, kernelSize: 7, padding: 3)
        var blocks = [TadaDecoderBlock]()
        for (i, stride) in rates.enumerated() {
            let inDim = channels / (1 << i)
            let outDim = channels / (1 << (i + 1))
            blocks.append(TadaDecoderBlock(inputDim: inDim, outputDim: outDim, stride: stride))
        }
        self.blocks = blocks
        let finalDim = channels / (1 << rates.count)
        self._finalSnake.wrappedValue = TadaSnake1d(channels: finalDim)
        self._finalConv.wrappedValue = Conv1d(inputChannels: finalDim, outputChannels: dOut, kernelSize: 7, padding: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = initialConv(x)
        for blk in blocks { out = blk(out) }
        out = finalSnake(out)
        out = finalConv(out)
        return tanh(out)
    }
}

// MARK: - Decoder (top-level)

class TadaDecoder: Module {
    @ModuleInfo(key: "decoder_proj") var decoderProj: Linear
    @ModuleInfo(key: "local_attention_decoder") var localAttentionDecoder: TadaLocalAttentionEncoder
    @ModuleInfo(key: "wav_decoder") var wavDecoder: TadaDACDecoder

    override init() {
        self._decoderProj.wrappedValue = Linear(kEmbedDim, kHiddenDim)
        self._localAttentionDecoder.wrappedValue = TadaLocalAttentionEncoder(
            dModel: kHiddenDim, numLayers: kNumAttnLayers, numHeads: kNumAttnHeads, dFf: kAttnDimFeedforward
        )
        self._wavDecoder.wrappedValue = TadaDACDecoder(inputChannel: kHiddenDim, channels: kWavDecoderChannels, rates: kDecoderStrides)
    }

    func callAsFunction(_ encodedExpanded: MLXArray, tokenMasks: MLXArray) -> MLXArray {
        var x = decoderProj(encodedExpanded)
        let attnMask = decoderCreateSegmentAttentionMask(tokenMasks)
        x = localAttentionDecoder(x, mask: attnMask)
        return wavDecoder(x)
    }

    func decodeFrames(encoded: MLXArray, timeBefore: MLXArray) -> MLXArray {
        let T = encoded.shape[0]
        guard T > 0 else { return MLXArray.zeros([0]) }

        let tbSlice = timeBefore[..<(T + 1)]
        var parts = [MLXArray]()

        for pos in 0..<T {
            let nZeros = max(0, tbSlice[pos].item(Int.self) - 1)
            if nZeros > 0 {
                parts.append(MLXArray.zeros([nZeros, encoded.shape[encoded.ndim - 1]]))
            }
            parts.append(expandedDimensions(encoded[pos], axis: 0))
        }

        let nTrailing = tbSlice[T].item(Int.self)
        if nTrailing > 0 {
            parts.append(MLXArray.zeros([nTrailing, encoded.shape[encoded.ndim - 1]]))
        }

        let expanded = expandedDimensions(concatenated(parts, axis: 0), axis: 0)
        let tokenMasks = (sqrt(sum(expanded * expanded, axis: -1)) .!= 0).asType(.int32)
        let wav = self(expanded, tokenMasks: tokenMasks)
        eval(wav)
        return wav.reshaped(-1)
    }
}
