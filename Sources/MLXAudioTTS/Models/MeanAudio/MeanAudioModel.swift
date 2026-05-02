//
//  MeanAudioModel.swift
//  MLXAudio
//
//  Main MeanAudio flow transformer (Flux-style MMDiT with dual timestep embeddings).
//  Ported from MeanAudio Python: meanaudio/model/networks.py (class MeanAudio)
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Preprocessed Conditions

struct MAPreprocessedConditions {
    var textF: MLXArray
    var textFC: MLXArray
}

// MARK: - Sequential helpers (for nn.Sequential equivalent)

class MAAudioInputProj: Module {
    @ModuleInfo var conv1: MAChannelLastConv1d
    @ModuleInfo var mlp: MAConvMLP

    init(latentDim: Int, hiddenDim: Int) {
        self._conv1.wrappedValue = MAChannelLastConv1d(
            inChannels: latentDim, outChannels: hiddenDim, kernelSize: 7, padding: 3
        )
        self._mlp.wrappedValue = MAConvMLP(
            dim: hiddenDim, hiddenDim: hiddenDim * 4, kernelSize: 7, padding: 3
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp(selu(conv1(x)))
    }
}

class MATextInputProj: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo var mlp: MAMLP

    init(textDim: Int, hiddenDim: Int) {
        self._linear.wrappedValue = Linear(textDim, hiddenDim)
        self._mlp.wrappedValue = MAMLP(dim: hiddenDim, hiddenDim: hiddenDim * 4)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp(linear(x))
    }
}

class MATextCondProj: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo var mlp: MAMLP

    init(textCDim: Int, hiddenDim: Int) {
        self._linear.wrappedValue = Linear(textCDim, hiddenDim)
        self._mlp.wrappedValue = MAMLP(dim: hiddenDim, hiddenDim: hiddenDim * 4)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp(linear(x))
    }
}

// MARK: - MeanAudio Model

class MeanAudioFlowTransformer: Module {
    let config: MeanAudioConfig

    @ModuleInfo(key: "audio_input_proj") var audioInputProj: MAAudioInputProj
    @ModuleInfo(key: "text_input_proj") var textInputProj: MATextInputProj
    @ModuleInfo(key: "text_cond_proj") var textCondProj: MATextCondProj
    @ModuleInfo(key: "final_layer") var finalLayer: MAFinalBlock
    @ModuleInfo(key: "t_embed") var tEmbed: MATimestepEmbedder
    @ModuleInfo(key: "r_embed") var rEmbed: MATimestepEmbedder
    @ModuleInfo(key: "joint_blocks") var jointBlocks: [MAJointBlock]
    @ModuleInfo(key: "fused_blocks") var fusedBlocks: [MAMMDitSingleBlock]

    let latentMean: MLXArray
    let latentStd: MLXArray
    let emptyStringFeat: MLXArray
    let emptyStringFeatC: MLXArray

    var latentRot: MLXArray?
    var textRot: MLXArray?

    init(config: MeanAudioConfig) {
        self.config = config

        let hiddenDim = config.hiddenDim
        let mmDepth = config.mmDepth

        self._audioInputProj.wrappedValue = MAAudioInputProj(
            latentDim: config.latentDim, hiddenDim: hiddenDim
        )
        self._textInputProj.wrappedValue = MATextInputProj(
            textDim: config.textDim, hiddenDim: hiddenDim
        )
        self._textCondProj.wrappedValue = MATextCondProj(
            textCDim: config.textCDim, hiddenDim: hiddenDim
        )

        self._finalLayer.wrappedValue = MAFinalBlock(dim: hiddenDim, outDim: config.latentDim)

        self._tEmbed.wrappedValue = MATimestepEmbedder(
            dim: hiddenDim, frequencyEmbeddingSize: 256, maxPeriod: 10000
        )
        self._rEmbed.wrappedValue = MATimestepEmbedder(
            dim: hiddenDim, frequencyEmbeddingSize: 256, maxPeriod: 10000
        )

        self._jointBlocks.wrappedValue = (0 ..< mmDepth).map { i in
            MAJointBlock(
                dim: hiddenDim,
                nhead: config.numHeads,
                mlpRatio: config.mlpRatio,
                preOnly: (i == mmDepth - 1)
            )
        }

        self._fusedBlocks.wrappedValue = (0 ..< config.fusedDepth).map { _ in
            MAMMDitSingleBlock(
                dim: hiddenDim,
                nhead: config.numHeads,
                mlpRatio: config.mlpRatio,
                kernelSize: 3,
                padding: 1
            )
        }

        // Placeholders — loaded from checkpoint
        self.latentMean = MLXArray.ones([1, 1, config.latentDim]) * Float.nan
        self.latentStd = MLXArray.ones([1, 1, config.latentDim]) * Float.nan
        self.emptyStringFeat = MLXArray.zeros([config.textSeqLen, config.textDim])
        self.emptyStringFeatC = MLXArray.zeros([config.textCDim])

        if config.useRope {
            self.latentRot = MARoPE.computeRotations(
                length: config.latentSeqLen,
                dim: config.headDim
            )
            self.textRot = MARoPE.computeRotations(
                length: config.textSeqLen,
                dim: config.headDim
            )
        } else {
            self.latentRot = nil
            self.textRot = nil
        }

        super.init()
    }

    func normalize(_ x: MLXArray) -> MLXArray {
        (x - latentMean) / latentStd
    }

    func unnormalize(_ x: MLXArray) -> MLXArray {
        x * latentStd + latentMean
    }

    func preprocessConditions(textF: MLXArray, textFC: MLXArray) -> MAPreprocessedConditions {
        let projectedFC = textCondProj(textFC)
        let projectedF = textInputProj(textF)
        return MAPreprocessedConditions(textF: projectedF, textFC: projectedFC)
    }

    func predictFlow(
        latent: MLXArray, t: MLXArray, r: MLXArray,
        conditions: MAPreprocessedConditions
    ) -> MLXArray {
        let textF = conditions.textF
        let textFC = conditions.textFC

        var lat = audioInputProj(latent)

        // global_c = t_embed(t) + r_embed(r) + text_f_c, all unsqueezed to (B, 1, D)
        let globalC = tEmbed(t).expandedDimensions(axis: 1)
            + rEmbed(r).expandedDimensions(axis: 1)
            + textFC.expandedDimensions(axis: 1)

        let extendedC = globalC

        var tf = textF
        for block in jointBlocks {
            (lat, tf) = block(
                latent: lat, textF: tf,
                globalC: globalC, extendedC: extendedC,
                latentRot: latentRot, textRot: textRot
            )
        }

        for block in fusedBlocks {
            lat = block(lat, cond: extendedC, rot: latentRot)
        }

        return finalLayer(lat, c: extendedC)
    }

    func getEmptyConditions(batchSize: Int) -> MAPreprocessedConditions {
        let emptyF = MLX.broadcast(
            emptyStringFeat.expandedDimensions(axis: 0),
            to: [1, config.textSeqLen, config.textDim]
        )
        let emptyFC = MLX.broadcast(
            emptyStringFeatC.expandedDimensions(axis: 0),
            to: [1, config.textCDim]
        )

        var conditions = preprocessConditions(textF: emptyF, textFC: emptyFC)
        conditions.textF = MLX.broadcast(
            conditions.textF,
            to: [batchSize, config.textSeqLen, config.hiddenDim]
        )
        conditions.textFC = MLX.broadcast(
            conditions.textFC,
            to: [batchSize, config.hiddenDim]
        )
        return conditions
    }

    func odeWrapper(
        t: MLXArray, r: MLXArray, latent: MLXArray,
        conditions: MAPreprocessedConditions,
        emptyConditions: MAPreprocessedConditions,
        cfgStrength: Float
    ) -> MLXArray {
        let bs = latent.dim(0)
        let tBatch = MLX.broadcast(t, to: [bs])
        let rBatch = MLX.broadcast(r, to: [bs])

        if cfgStrength < 1.0 {
            return predictFlow(latent: latent, t: tBatch, r: rBatch, conditions: conditions)
        } else {
            let condFlow = predictFlow(
                latent: latent, t: tBatch, r: rBatch, conditions: conditions
            )
            let uncondFlow = predictFlow(
                latent: latent, t: tBatch, r: rBatch, conditions: emptyConditions
            )
            return cfgStrength * condFlow + (1 - cfgStrength) * uncondFlow
        }
    }
}
