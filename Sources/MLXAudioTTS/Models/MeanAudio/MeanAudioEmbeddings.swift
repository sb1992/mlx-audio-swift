//
//  MeanAudioEmbeddings.swift
//  MLXAudio
//
//  Timestep embedding and RoPE for MeanAudio.
//  Ported from MeanAudio Python: meanaudio/model/embeddings.py
//  and meanaudio/ext/rotary_embeddings.py
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Timestep Embedder

class MATimestepEmbedder: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    let freqs: MLXArray

    init(dim: Int, frequencyEmbeddingSize: Int = 256, maxPeriod: Int = 10000) {
        let halfDim = frequencyEmbeddingSize / 2
        let baseFreqs = MLXArray(1.0) / MLX.pow(
            MLXArray(Float(10000)),
            MLXArray(stride(from: 0, to: frequencyEmbeddingSize, by: 2).map { Float($0) / Float(frequencyEmbeddingSize) })
        )
        let freqScale = Float(10000) / Float(maxPeriod)
        self.freqs = freqScale * baseFreqs

        self._linear1.wrappedValue = Linear(frequencyEmbeddingSize, dim)
        self._linear2.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func timestepEmbedding(_ t: MLXArray) -> MLXArray {
        let args = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = timestepEmbedding(t).asType(t.dtype)
        return linear2(silu(linear1(tFreq)))
    }
}

// MARK: - Rotary Position Embeddings

enum MARoPE {
    static func computeRotations(length: Int, dim: Int, theta: Int = 10000, freqScaling: Float = 1.0) -> MLXArray {
        let pos = MLXArray(0 ..< length).asType(.float32)
        let freqs = MLXArray(1.0) / MLX.pow(
            MLXArray(Float(theta)),
            MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        ) * MLXArray(freqScaling)

        // pos: (length,), freqs: (dim/2,) → outer product → (length, dim/2)
        let rot = pos.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)

        // Build 2x2 rotation matrices: (length, dim/2, 2, 2)
        let cosRot = MLX.cos(rot)
        let sinRot = MLX.sin(rot)
        let negSinRot = -sinRot

        // Stack as [cos, -sin, sin, cos] then reshape to (1, length, dim/2, 2, 2)
        let row0 = MLX.stacked([cosRot, negSinRot], axis: -1)   // (length, dim/2, 2)
        let row1 = MLX.stacked([sinRot, cosRot], axis: -1)      // (length, dim/2, 2)
        let rotMatrix = MLX.stacked([row0, row1], axis: -2)      // (length, dim/2, 2, 2)

        return rotMatrix.expandedDimensions(axis: 0)  // (1, length, dim/2, 2, 2)
    }

    static func applyRoPE(_ x: MLXArray, rot: MLXArray) -> MLXArray {
        // x: (B, heads, seq, headDim)
        let xFloat = x.asType(.float32)
        let shape = xFloat.shape
        // Reshape to (..., dim/2, 1, 2)
        let xReshaped = xFloat.reshaped(shape.dropLast() + [shape.last! / 2, 1, 2])

        // rot: (1, seq, dim/2, 2, 2) — matrix multiply via element-wise + sum
        // result[..., 0] = rot[..., 0, 0] * x[..., 0] + rot[..., 0, 1] * x[..., 1]
        // result[..., 1] = rot[..., 1, 0] * x[..., 0] + rot[..., 1, 1] * x[..., 1]
        let x0 = xReshaped[.ellipsis, 0]  // (..., dim/2, 1)
        let x1 = xReshaped[.ellipsis, 1]  // (..., dim/2, 1)

        let out0 = rot[.ellipsis, 0, 0] * x0 + rot[.ellipsis, 0, 1] * x1
        let out1 = rot[.ellipsis, 1, 0] * x0 + rot[.ellipsis, 1, 1] * x1

        let xOut = MLX.stacked([out0, out1], axis: -1)
        return xOut.reshaped(shape).asType(x.dtype)
    }
}
