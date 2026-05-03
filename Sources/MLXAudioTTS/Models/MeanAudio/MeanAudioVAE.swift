//
//  MeanAudioVAE.swift
//  MLXAudio
//
//  EDM2-style magnitude-preserving VAE decoder for MeanAudio.
//  Converts latent z → mel spectrogram. Only decoder path needed for inference.
//  Ported from MeanAudio Python: meanaudio/ext/autoencoder/vae.py,
//  vae_modules.py, edm2_utils.py
//

import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

// MARK: - EDM2 Utilities

private func mpNormalize(_ x: MLXArray, dim: [Int]? = nil, eps: Float = 1e-4) -> MLXArray {
    let dims = dim ?? Array(1 ..< x.ndim)
    let norm = MLX.sqrt(x.square().sum(axes: dims, keepDims: true).asType(.float32))
    let factor = Float(sqrt(Double(norm.size) / Double(x.size)))
    let denom = (MLXArray(eps) + factor * norm).asType(x.dtype)
    return x / denom
}

private func mpSilu(_ x: MLXArray) -> MLXArray {
    silu(x) / MLXArray(Float(0.596))
}

private func mpSum(_ a: MLXArray, _ b: MLXArray, t: Float = 0.5) -> MLXArray {
    let result = a + MLXArray(t) * (b - a)  // lerp
    let scale = Float(1.0 / sqrt(Double((1 - t) * (1 - t) + t * t)))
    return result * MLXArray(scale)
}

// MARK: - Magnitude-Preserving Conv1d (EDM2)

class MAMPConv1D: Module {
    var weight: MLXArray
    let outChannels: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int) {
        self.outChannels = outChannels
        // Weight shape for MLX Conv1d: (out, kernel, in)
        self.weight = MLXRandom.normal([outChannels, kernelSize, inChannels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gain: MLXArray? = nil) -> MLXArray {
        if weight.ndim == 2 {
            var w = weight / (MLX.sqrt(weight.square().sum(axis: 1, keepDims: true)) + 1e-8)
            if let gain { w = w * gain }
            return MLX.matmul(x, w.transposed())
        }
        var w = weight / (MLX.sqrt(weight.square().sum(axes: [1, 2], keepDims: true)) + 1e-8)
        if let gain { w = w * gain }
        let padding = w.dim(1) / 2
        let xCL = x.transposed(0, 2, 1)
        let out = MLX.conv1d(xCL, w, padding: padding)
        return out.transposed(0, 2, 1)
    }
}

// MARK: - Resnet Block 1D

class MAResnetBlock1D: Module {
    let inDim: Int
    let outDim: Int
    let useNorm: Bool
    let useConvShortcut: Bool

    @ModuleInfo var conv1: MAMPConv1D
    @ModuleInfo var conv2: MAMPConv1D
    @ModuleInfo(key: "conv_shortcut") var convShortcut: MAMPConv1D?
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: MAMPConv1D?

    init(inDim: Int, outDim: Int? = nil, convShortcut: Bool = false, kernelSize: Int = 3, useNorm: Bool = true) {
        let out = outDim ?? inDim
        self.inDim = inDim
        self.outDim = out
        self.useNorm = useNorm
        self.useConvShortcut = convShortcut

        self._conv1.wrappedValue = MAMPConv1D(inChannels: inDim, outChannels: out, kernelSize: kernelSize)
        self._conv2.wrappedValue = MAMPConv1D(inChannels: out, outChannels: out, kernelSize: kernelSize)

        if inDim != out {
            if convShortcut {
                self._convShortcut.wrappedValue = MAMPConv1D(inChannels: inDim, outChannels: out, kernelSize: kernelSize)
                self._ninShortcut.wrappedValue = nil
            } else {
                self._convShortcut.wrappedValue = nil
                self._ninShortcut.wrappedValue = MAMPConv1D(inChannels: inDim, outChannels: out, kernelSize: 1)
            }
        } else {
            self._convShortcut.wrappedValue = nil
            self._ninShortcut.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var inp = x
        if useNorm {
            inp = mpNormalize(inp, dim: [1])
        }

        var h = mpSilu(inp)
        h = conv1(h)
        h = mpSilu(h)
        h = conv2(h)

        if inDim != outDim {
            if useConvShortcut, let cs = convShortcut {
                inp = cs(inp)
            } else if let ns = ninShortcut {
                inp = ns(inp)
            }
        }

        return mpSum(inp, h, t: 0.3)
    }
}

// MARK: - Attention Block 1D

class MAAttnBlock1D: Module {
    let inChannels: Int
    let numHeads: Int

    @ModuleInfo var qkv: MAMPConv1D
    @ModuleInfo(key: "proj_out") var projOut: MAMPConv1D

    init(inChannels: Int, numHeads: Int = 1) {
        self.inChannels = inChannels
        self.numHeads = numHeads

        self._qkv.wrappedValue = MAMPConv1D(inChannels: inChannels, outChannels: inChannels * 3, kernelSize: 1)
        self._projOut.wrappedValue = MAMPConv1D(inChannels: inChannels, outChannels: inChannels, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let C = inChannels
        let T = x.dim(2)
        let headDim = C / numHeads

        let y = qkv(x)
        // y: (B, 3*C, T) → (B, heads, headDim, 3, T)
        let yReshaped = y.reshaped(B, numHeads, headDim, 3, T)

        // Normalize along channel dim (dim=2)
        let yNorm = mpNormalize(yReshaped, dim: [2])

        // Split q, k, v along dim=3
        let q = yNorm[0..., 0..., 0..., 0, 0...].transposed(0, 1, 3, 2)  // (B, H, T, D)
        let k = yNorm[0..., 0..., 0..., 1, 0...].transposed(0, 1, 3, 2)
        let v = yNorm[0..., 0..., 0..., 2, 0...].transposed(0, 1, 3, 2)

        let scale = 1.0 / sqrt(Float(headDim))
        let h = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        // h: (B, H, T, D) → (B, H*D, T)
        let hOut = h.transposed(0, 1, 3, 2).reshaped(B, C, T)

        let projected = projOut(hOut)
        return mpSum(x, projected, t: 0.3)
    }
}

// MARK: - Upsample 1D

class MAUpsample1D: Module {
    let withConv: Bool
    @ModuleInfo var conv: MAMPConv1D?

    init(inChannels: Int, withConv: Bool) {
        self.withConv = withConv
        if withConv {
            self._conv.wrappedValue = MAMPConv1D(inChannels: inChannels, outChannels: inChannels, kernelSize: 3)
        } else {
            self._conv.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Nearest-neighbor upsample 2x along last dim
        // x: (B, C, T) → (B, C, 2*T)
        let repeated = MLX.repeated(x, count: 2, axis: 2)
        let B = x.dim(0)
        let C = x.dim(1)
        let T = x.dim(2)
        // Interleave: expand and reshape
        let expanded = x.expandedDimensions(axis: 3)  // (B, C, T, 1)
        let tiled = MLX.concatenated([expanded, expanded], axis: 3)  // (B, C, T, 2)
        var upsampled = tiled.reshaped(B, C, T * 2)

        if withConv, let conv {
            upsampled = conv(upsampled)
        }
        return upsampled
    }
}

// MARK: - Decoder Mid Block

class MADecoderMid: Module {
    @ModuleInfo(key: "block_1") var block1: MAResnetBlock1D
    @ModuleInfo(key: "attn_1") var attn1: MAAttnBlock1D
    @ModuleInfo(key: "block_2") var block2: MAResnetBlock1D

    init(blockIn: Int) {
        self._block1.wrappedValue = MAResnetBlock1D(inDim: blockIn, outDim: blockIn, useNorm: true)
        self._attn1.wrappedValue = MAAttnBlock1D(inChannels: blockIn)
        self._block2.wrappedValue = MAResnetBlock1D(inDim: blockIn, outDim: blockIn, useNorm: true)
        super.init()
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        var out = block1(h)
        out = attn1(out)
        out = block2(out)
        return out
    }
}

// MARK: - Decoder Up Level

class MADecoderUpLevel: Module {
    @ModuleInfo var block: [MAResnetBlock1D]
    @ModuleInfo var attn: [MAAttnBlock1D]
    @ModuleInfo var upsample: MAUpsample1D?

    init(blockIn: Int, blockOut: Int, numResBlocks: Int, hasAttn: Bool, hasUpsample: Bool) {
        var blocks: [MAResnetBlock1D] = []
        var attns: [MAAttnBlock1D] = []
        var currentIn = blockIn

        for _ in 0 ..< (numResBlocks + 1) {
            blocks.append(MAResnetBlock1D(inDim: currentIn, outDim: blockOut, useNorm: true))
            currentIn = blockOut
            if hasAttn {
                attns.append(MAAttnBlock1D(inChannels: blockOut))
            }
        }

        self._block.wrappedValue = blocks
        self._attn.wrappedValue = attns

        if hasUpsample {
            self._upsample.wrappedValue = MAUpsample1D(inChannels: blockOut, withConv: true)
        } else {
            self._upsample.wrappedValue = nil
        }
        super.init()
    }
}

// MARK: - VAE Decoder

class MAVAEDecoder: Module {
    let numLayers: Int
    let numResBlocks: Int
    let clipAct: Float
    let downLayers: [Int]

    @ModuleInfo(key: "conv_in") var convIn: MAMPConv1D
    @ModuleInfo var mid: MADecoderMid
    @ModuleInfo var up: [MADecoderUpLevel]
    @ModuleInfo(key: "conv_out") var convOut: MAMPConv1D
    var learnableGain: MLXArray

    init(
        dim: Int = 384,
        outDim: Int = 80,
        chMult: [Int] = [1, 2, 4],
        numResBlocks: Int = 2,
        attnLayers: [Int] = [3],
        downLayers: [Int] = [0],
        embedDim: Int = 20,
        clipAct: Float = 256.0
    ) {
        self.numLayers = chMult.count
        self.numResBlocks = numResBlocks
        self.clipAct = clipAct
        self.downLayers = downLayers.map { $0 + 1 }

        let blockIn = dim * chMult[chMult.count - 1]

        self._convIn.wrappedValue = MAMPConv1D(inChannels: embedDim, outChannels: blockIn, kernelSize: 3)
        self._mid.wrappedValue = MADecoderMid(blockIn: blockIn)

        // Build up levels in reverse order
        var levels: [MADecoderUpLevel] = []
        var currentIn = blockIn
        for iLevel in stride(from: chMult.count - 1, through: 0, by: -1) {
            let blockOut = dim * chMult[iLevel]
            let hasAttn = attnLayers.contains(iLevel)
            let hasUpsample = self.downLayers.contains(iLevel)

            levels.append(MADecoderUpLevel(
                blockIn: currentIn,
                blockOut: blockOut,
                numResBlocks: numResBlocks,
                hasAttn: hasAttn,
                hasUpsample: hasUpsample
            ))
            currentIn = blockOut
        }
        // Reverse to match Python's insert(0, ...) behavior
        self._up.wrappedValue = levels.reversed()

        self._convOut.wrappedValue = MAMPConv1D(inChannels: currentIn, outChannels: outDim, kernelSize: 3)
        self.learnableGain = MLXArray(Float(0))

        super.init()
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = mid(h)
        h = clip(h, min: -clipAct, max: clipAct)

        for iLevel in stride(from: numLayers - 1, through: 0, by: -1) {
            let level = up[iLevel]
            for iBlock in 0 ..< (numResBlocks + 1) {
                h = level.block[iBlock](h)
                if !level.attn.isEmpty {
                    h = level.attn[iBlock](h)
                }
                h = clip(h, min: -clipAct, max: clipAct)
            }
            if let upsample = level.upsample {
                h = upsample(h)
            }
        }

        h = mpSilu(h)
        h = convOut(h, gain: learnableGain + MLXArray(Float(1)))
        return h
    }
}

// MARK: - Full VAE (decode-only for inference)

class MeanAudioVAE: Module {
    var dataMean: MLXArray
    var dataStd: MLXArray

    @ModuleInfo var decoder: MAVAEDecoder

    init(dataDim: Int = 80, embedDim: Int = 20, hiddenDim: Int = 384) {
        let mean: [Float]
        let std: [Float]

        if dataDim == 80 {
            mean = MeanAudioVAEConstants.dataMean80D
            std = MeanAudioVAEConstants.dataStd80D
        } else {
            mean = MeanAudioVAEConstants.dataMean128D
            std = MeanAudioVAEConstants.dataStd128D
        }

        self.dataMean = MLXArray(mean).reshaped(1, dataDim, 1)
        self.dataStd = MLXArray(std).reshaped(1, dataDim, 1)

        self._decoder.wrappedValue = MAVAEDecoder(
            dim: hiddenDim, outDim: dataDim, embedDim: embedDim
        )
        super.init()
    }

    func decode(_ z: MLXArray) -> MLXArray {
        var mel = decoder(z)
        mel = mel * dataStd + dataMean
        return mel
    }
}

// MARK: - Constants

private enum MeanAudioVAEConstants {
    static let dataMean80D: [Float] = [
        -1.6058, -1.3676, -1.2520, -1.2453, -1.2078, -1.2224, -1.2419, -1.2439, -1.2922, -1.2927,
        -1.3170, -1.3543, -1.3401, -1.3836, -1.3907, -1.3912, -1.4313, -1.4152, -1.4527, -1.4728,
        -1.4568, -1.5101, -1.5051, -1.5172, -1.5623, -1.5373, -1.5746, -1.5687, -1.6032, -1.6131,
        -1.6081, -1.6331, -1.6489, -1.6489, -1.6700, -1.6738, -1.6953, -1.6969, -1.7048, -1.7280,
        -1.7361, -1.7495, -1.7658, -1.7814, -1.7889, -1.8064, -1.8221, -1.8377, -1.8417, -1.8643,
        -1.8857, -1.8929, -1.9173, -1.9379, -1.9531, -1.9673, -1.9824, -2.0042, -2.0215, -2.0436,
        -2.0766, -2.1064, -2.1418, -2.1855, -2.2319, -2.2767, -2.3161, -2.3572, -2.3954, -2.4282,
        -2.4659, -2.5072, -2.5552, -2.6074, -2.6584, -2.7107, -2.7634, -2.8266, -2.8981, -2.9673,
    ]

    static let dataStd80D: [Float] = [
        1.0291, 1.0411, 1.0043, 0.9820, 0.9677, 0.9543, 0.9450, 0.9392, 0.9343, 0.9297, 0.9276, 0.9263,
        0.9242, 0.9254, 0.9232, 0.9281, 0.9263, 0.9315, 0.9274, 0.9247, 0.9277, 0.9199, 0.9188, 0.9194,
        0.9160, 0.9161, 0.9146, 0.9161, 0.9100, 0.9095, 0.9145, 0.9076, 0.9066, 0.9095, 0.9032, 0.9043,
        0.9038, 0.9011, 0.9019, 0.9010, 0.8984, 0.8983, 0.8986, 0.8961, 0.8962, 0.8978, 0.8962, 0.8973,
        0.8993, 0.8976, 0.8995, 0.9016, 0.8982, 0.8972, 0.8974, 0.8949, 0.8940, 0.8947, 0.8936, 0.8939,
        0.8951, 0.8956, 0.9017, 0.9167, 0.9436, 0.9690, 1.0003, 1.0225, 1.0381, 1.0491, 1.0545, 1.0604,
        1.0761, 1.0929, 1.1089, 1.1196, 1.1176, 1.1156, 1.1117, 1.1070,
    ]

    static let dataMean128D: [Float] = [
        -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006, -2.2357, -2.4597,
        -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047, -2.7483, -2.5926, -2.7462, -2.7033,
        -2.7386, -2.8112, -2.7502, -2.9594, -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157,
        -3.1191, -2.9893, -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
        -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509, -3.5089, -3.4647,
        -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747, -3.7072, -3.7279, -3.7283, -3.7795,
        -3.8259, -3.8447, -3.8663, -3.9182, -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121,
        -4.1488, -4.1874, -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
        -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053, -5.4927, -5.5712,
        -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103, -6.0955, -6.1673, -6.2362, -6.3120,
        -6.3926, -6.4797, -6.5565, -6.6511, -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663,
        -7.6136, -7.7469, -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
        -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861,
    ]

    static let dataStd128D: [Float] = [
        2.3804, 2.4368, 2.3772, 2.3145, 2.2803, 2.2510, 2.2316, 2.2083, 2.1996, 2.1835, 2.1769, 2.1659,
        2.1631, 2.1618, 2.1540, 2.1606, 2.1571, 2.1567, 2.1612, 2.1579, 2.1679, 2.1683, 2.1634, 2.1557,
        2.1668, 2.1518, 2.1415, 2.1449, 2.1406, 2.1350, 2.1313, 2.1415, 2.1281, 2.1352, 2.1219, 2.1182,
        2.1327, 2.1195, 2.1137, 2.1080, 2.1179, 2.1036, 2.1087, 2.1036, 2.1015, 2.1068, 2.0975, 2.0991,
        2.0902, 2.1015, 2.0857, 2.0920, 2.0893, 2.0897, 2.0910, 2.0881, 2.0925, 2.0873, 2.0960, 2.0900,
        2.0957, 2.0958, 2.0978, 2.0936, 2.0886, 2.0905, 2.0845, 2.0855, 2.0796, 2.0840, 2.0813, 2.0817,
        2.0838, 2.0840, 2.0917, 2.1061, 2.1431, 2.1976, 2.2482, 2.3055, 2.3700, 2.4088, 2.4372, 2.4609,
        2.4731, 2.4847, 2.5072, 2.5451, 2.5772, 2.6147, 2.6529, 2.6596, 2.6645, 2.6726, 2.6803, 2.6812,
        2.6899, 2.6916, 2.6931, 2.6998, 2.7062, 2.7262, 2.7222, 2.7158, 2.7041, 2.7485, 2.7491, 2.7451,
        2.7485, 2.7233, 2.7297, 2.7233, 2.7145, 2.6958, 2.6788, 2.6439, 2.6007, 2.4786, 2.2469, 2.1877,
        2.1392, 2.0717, 2.0107, 1.9676, 1.9140, 1.7102, 0.9101, 0.7164,
    ]
}
