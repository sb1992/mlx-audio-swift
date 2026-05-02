import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Constants

private let kHiddenSize = 1024
private let kNumAttentionHeads = 16
private let kIntermediateSize = 4096
private let kNumHiddenLayers = 24
private let kConvDim = [512, 512, 512, 512, 512, 512, 512]
private let kConvKernel = [10, 3, 3, 3, 3, 2, 2]
private let kConvStride = [5, 2, 2, 2, 2, 2, 2]
private let kNumConvPosEmbeddings = 128
private let kNumConvPosEmbeddingGroups = 16
private let kVocabSize = 128256

// MARK: - Viterbi Alignment

func tadaAlignTextTokens(probs: [[Float]], textTokens: [Int]) -> [Int] {
    let L = probs.count
    let T = textTokens.count
    guard L > 0 && T > 0 else { return Array(repeating: 0, count: T) }

    var tokenProbs = [[Float]](repeating: [Float](repeating: 0, count: T), count: L)
    for i in 0..<L {
        for j in 0..<T {
            tokenProbs[i][j] = probs[i][textTokens[j]]
        }
    }

    var F = [[Float]](repeating: [Float](repeating: -.infinity, count: T), count: L)
    var backpointer = [[Int]](repeating: [Int](repeating: 0, count: T), count: L)

    var cumMaxVal = tokenProbs[0][0]
    var cumMaxIdx = 0
    F[0][0] = cumMaxVal
    backpointer[0][0] = 0

    for i in 1..<L {
        if tokenProbs[i][0] > cumMaxVal {
            cumMaxVal = tokenProbs[i][0]
            cumMaxIdx = i
        }
        F[i][0] = cumMaxVal
        backpointer[i][0] = cumMaxIdx
    }

    if T <= L {
        var cumsum: Float = 0
        for k in 0..<T {
            cumsum += tokenProbs[k][k]
            F[k][k] = cumsum
            backpointer[k][k] = k
        }
    }

    for i in 1..<L {
        let maxJ = min(i, T)
        guard maxJ > 1 else { continue }
        for j in 1..<maxJ {
            let skipScore = F[i - 1][j]
            let useScore = F[i - 1][j - 1] + tokenProbs[i][j]
            if useScore >= skipScore {
                F[i][j] = useScore
                backpointer[i][j] = i
            } else {
                F[i][j] = skipScore
                backpointer[i][j] = -1
            }
        }
    }

    var positions = [Int](repeating: 0, count: T)
    var i = L - 1
    var j = T - 1

    while j >= 0 {
        if j == 0 {
            positions[0] = backpointer[i][j]
            break
        } else if backpointer[i][j] == -1 {
            i -= 1
        } else {
            positions[j] = backpointer[i][j]
            i -= 1
            j -= 1
        }
    }

    return positions
}

// MARK: - Group Norm

class TadaGroupNorm: Module {
    let numGroups: Int
    let numChannels: Int
    let eps: Float
    let affine: Bool
    var weight: MLXArray?
    var bias: MLXArray?

    init(numGroups: Int, numChannels: Int, eps: Float = 1e-5, affine: Bool = true) {
        self.numGroups = numGroups
        self.numChannels = numChannels
        self.eps = eps
        self.affine = affine
        self.weight = affine ? MLXArray.ones([numChannels]) : nil
        self.bias = affine ? MLXArray.zeros([numChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let C = x.shape[x.ndim - 1]
        let G = numGroups
        let spatial = Array(x.shape[1..<(x.ndim - 1)])

        var out = x.reshaped(B, -1, G, C / G)
        out = out.transposed(0, 2, 1, 3)
        out = out.reshaped(B, G, -1)
        let mean = MLX.mean(out, axis: -1, keepDims: true)
        let variance = MLX.mean(pow(out - mean, 2), axis: -1, keepDims: true)
        out = (out - mean) * rsqrt(variance + eps)
        out = out.reshaped(B, G, -1, C / G)
        out = out.transposed(0, 2, 1, 3)

        let shape = [B] + spatial + [C]
        out = out.reshaped(shape)

        if affine, let w = weight, let b = bias {
            out = out * w + b
        }
        return out
    }
}

// MARK: - Wav2Vec2 Feature Extractor Layer

class TadaWav2Vec2FeatureExtractorLayer: Module {
    var conv: Conv1d
    let isFirst: Bool
    @ModuleInfo(key: "layer_norm") var layerNorm: TadaGroupNorm?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, isFirst: Bool = false) {
        self.conv = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize, stride: stride, padding: 0, bias: false)
        self.isFirst = isFirst
        self._layerNorm.wrappedValue = isFirst ? TadaGroupNorm(numGroups: outChannels, numChannels: outChannels) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        if isFirst, let ln = layerNorm { out = ln(out) }
        return gelu(out)
    }
}

// MARK: - Wav2Vec2 Feature Extractor

class TadaWav2Vec2FeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [TadaWav2Vec2FeatureExtractorLayer]

    override init() {
        var layers = [TadaWav2Vec2FeatureExtractorLayer]()
        var inCh = 1
        for i in 0..<kConvDim.count {
            layers.append(TadaWav2Vec2FeatureExtractorLayer(
                inChannels: inCh, outChannels: kConvDim[i],
                kernelSize: kConvKernel[i], stride: kConvStride[i],
                isFirst: i == 0
            ))
            inCh = kConvDim[i]
        }
        self._convLayers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in convLayers { out = layer(out) }
        return out
    }
}

// MARK: - Wav2Vec2 Feature Projection

class TadaWav2Vec2FeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    var projection: Linear

    override init() {
        self._layerNorm.wrappedValue = LayerNorm(dimensions: kConvDim.last!)
        self.projection = Linear(kConvDim.last!, kHiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}

// MARK: - Wav2Vec2 Positional Conv Embedding

class TadaWav2Vec2PositionalConvEmbedding: Module {
    var conv: Conv1d

    override init() {
        self.conv = Conv1d(
            inputChannels: kHiddenSize, outputChannels: kHiddenSize,
            kernelSize: kNumConvPosEmbeddings, padding: kNumConvPosEmbeddings / 2,
            groups: kNumConvPosEmbeddingGroups
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(conv(x))
    }
}

// MARK: - Wav2Vec2 Attention

class TadaWav2Vec2Attention: Module {
    let numHeads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    override init() {
        self.numHeads = kNumAttentionHeads
        self.headDim = kHiddenSize / kNumAttentionHeads
        self._qProj.wrappedValue = Linear(kHiddenSize, kHiddenSize)
        self._kProj.wrappedValue = Linear(kHiddenSize, kHiddenSize)
        self._vProj.wrappedValue = Linear(kHiddenSize, kHiddenSize)
        self._outProj.wrappedValue = Linear(kHiddenSize, kHiddenSize)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])
        let q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let scale = pow(Float(headDim), -0.5)
        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        if let mask { attn = attn + mask }
        attn = softmax(attn, axis: -1)
        let out = attn.matmul(v).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return outProj(out)
    }
}

// MARK: - Wav2Vec2 Feed Forward

class TadaWav2Vec2FeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    override init() {
        self._intermediateDense.wrappedValue = Linear(kHiddenSize, kIntermediateSize)
        self._outputDense.wrappedValue = Linear(kIntermediateSize, kHiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }
}

// MARK: - Wav2Vec2 Encoder Layer

class TadaWav2Vec2EncoderLayer: Module {
    var attention: TadaWav2Vec2Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: TadaWav2Vec2FeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        self.attention = TadaWav2Vec2Attention()
        self._layerNorm.wrappedValue = LayerNorm(dimensions: kHiddenSize)
        self._feedForward.wrappedValue = TadaWav2Vec2FeedForward()
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: kHiddenSize)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = attention(x, mask: mask)
        out = layerNorm(x + out)
        let residual = out
        out = feedForward(out)
        return finalLayerNorm(residual + out)
    }
}

// MARK: - Wav2Vec2 Encoder

class TadaWav2Vec2Encoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: TadaWav2Vec2PositionalConvEmbedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    var layers: [TadaWav2Vec2EncoderLayer]

    override init() {
        self._posConvEmbed.wrappedValue = TadaWav2Vec2PositionalConvEmbedding()
        self._layerNorm.wrappedValue = LayerNorm(dimensions: kHiddenSize)
        self.layers = (0..<kNumHiddenLayers).map { _ in TadaWav2Vec2EncoderLayer() }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let pos = posConvEmbed(x)
        var out = x + pos[0..., ..<x.shape[1], 0...]
        out = layerNorm(out)
        for layer in layers { out = layer(out, mask: mask) }
        return out
    }
}

// MARK: - Wav2Vec2ForCTC

class TadaWav2Vec2ForCTC: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: TadaWav2Vec2FeatureExtractor
    @ModuleInfo(key: "feature_projection") var featureProjection: TadaWav2Vec2FeatureProjection
    var encoder: TadaWav2Vec2Encoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    override init() {
        self._featureExtractor.wrappedValue = TadaWav2Vec2FeatureExtractor()
        self._featureProjection.wrappedValue = TadaWav2Vec2FeatureProjection()
        self.encoder = TadaWav2Vec2Encoder()
        self._lmHead.wrappedValue = Linear(kHiddenSize, kVocabSize)
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var x = expandedDimensions(audio, axis: -1)
        x = featureExtractor(x)
        x = featureProjection(x)
        x = encoder(x)
        return lmHead(x)
    }
}

// MARK: - Aligner

class TadaAligner: Module {
    var wav2vec2: TadaWav2Vec2ForCTC

    override init() {
        self.wav2vec2 = TadaWav2Vec2ForCTC()
    }

    func align(
        audio16k: MLXArray,
        textTokens: [[Int]],
        inputLengths: [Int],
        eosTokenId: Int
    ) -> (positions: [[Int]], masks: [[Int]]) {
        let logits = wav2vec2(audio16k)
        let maxInputLength = inputLengths.max() ?? 0
        let batchSize = textTokens.count
        var allTokenPositions = [[Int]]()
        var allTokenMasks = [[Int]]()

        for batchIdx in 0..<batchSize {
            let batchTextTokens = textTokens[batchIdx]
            let filteredTokens = batchTextTokens.filter { $0 != eosTokenId }
            let validColumns = [0] + filteredTokens

            let logitsBatch = logits[batchIdx]
            let validColumnsArr = MLXArray(validColumns.map { Int32($0) })
            let logitsSparse = logitsBatch[0..., validColumnsArr].asType(.float32)
            eval(logitsSparse)

            let L = logitsSparse.shape[0]
            let V = logitsSparse.shape[1]
            var probs2d = [[Float]]()
            for row in 0..<L {
                var rowVals = [Float]()
                for col in 0..<V {
                    rowVals.append(logitsSparse[row, col].item(Float.self))
                }
                probs2d.append(rowVals)
            }

            var colToSparse = [Int: Int]()
            for (i, v) in validColumns.enumerated() { colToSparse[v] = i }
            let sparseTextTokens = filteredTokens.map { colToSparse[$0]! }

            let positions = tadaAlignTextTokens(probs: probs2d, textTokens: sparseTextTokens)

            var posEmb = [Int](repeating: 0, count: maxInputLength)
            for p in positions where p < maxInputLength { posEmb[p] = 1 }
            let positions1Indexed = positions.map { $0 + 1 }

            allTokenPositions.append(positions1Indexed)
            allTokenMasks.append(posEmb)
        }

        return (allTokenPositions, allTokenMasks)
    }
}
