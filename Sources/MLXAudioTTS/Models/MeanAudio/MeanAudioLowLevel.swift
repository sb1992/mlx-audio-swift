//
//  MeanAudioLowLevel.swift
//  MLXAudio
//
//  Low-level building blocks: ChannelLastConv1d, gated MLP, ConvMLP.
//  Ported from MeanAudio Python: meanaudio/model/low_level.py
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Channel-Last Conv1d

class MAChannelLastConv1d: Module {
    @ModuleInfo var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int = 1, padding: Int = 0, bias: Bool = true) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            padding: padding,
            bias: bias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, Seq, D) → transpose to (B, D, Seq) for Conv1d → back
        let transposed = x.transposed(0, 2, 1)
        let out = conv(transposed)
        return out.transposed(0, 2, 1)
    }
}

// MARK: - Gated MLP (LLaMA-style SwiGLU)

class MAMLP: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int, multipleOf: Int = 256) {
        var h = Int(2 * hiddenDim / 3)
        h = multipleOf * ((h + multipleOf - 1) / multipleOf)

        self._w1.wrappedValue = Linear(dim, h, bias: false)
        self._w2.wrappedValue = Linear(h, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, h, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Convolutional Gated MLP

class MAConvMLP: Module {
    @ModuleInfo var w1: MAChannelLastConv1d
    @ModuleInfo var w2: MAChannelLastConv1d
    @ModuleInfo var w3: MAChannelLastConv1d

    init(dim: Int, hiddenDim: Int, multipleOf: Int = 256, kernelSize: Int = 3, padding: Int = 1) {
        var h = Int(2 * hiddenDim / 3)
        h = multipleOf * ((h + multipleOf - 1) / multipleOf)

        self._w1.wrappedValue = MAChannelLastConv1d(inChannels: dim, outChannels: h, kernelSize: kernelSize, padding: padding, bias: false)
        self._w2.wrappedValue = MAChannelLastConv1d(inChannels: h, outChannels: dim, kernelSize: kernelSize, padding: padding, bias: false)
        self._w3.wrappedValue = MAChannelLastConv1d(inChannels: dim, outChannels: h, kernelSize: kernelSize, padding: padding, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}
