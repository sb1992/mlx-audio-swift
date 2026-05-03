//
//  MeanAudioUnitTests.swift
//  MLXAudio
//
//  Unit tests for MeanAudio model components — no weights needed.
//

import Testing
import MLX
@testable import MLXAudioTTS

@Suite("MeanAudio Unit Tests")
struct MeanAudioUnitTests {

    @Test func configDefaults() {
        let config = MeanAudioConfig.small
        #expect(config.latentDim == 20)
        #expect(config.hiddenDim == 448)
        #expect(config.numHeads == 7)
        #expect(config.headDim == 64)
        #expect(config.depth == 12)
        #expect(config.fusedDepth == 8)
        #expect(config.mmDepth == 4)
    }

    @Test func channelLastConv1dShape() {
        let conv = MAChannelLastConv1d(inChannels: 20, outChannels: 64, kernelSize: 7, padding: 3)
        let x = MLXRandom.normal([2, 312, 20])
        let out = conv(x)
        eval(out)
        #expect(out.shape == [2, 312, 64])
    }

    @Test func mlpShape() {
        let mlp = MAMLP(dim: 448, hiddenDim: 1792)
        let x = MLXRandom.normal([2, 312, 448])
        let out = mlp(x)
        eval(out)
        #expect(out.shape == [2, 312, 448])
    }

    @Test func convMlpShape() {
        let mlp = MAConvMLP(dim: 448, hiddenDim: 1792, kernelSize: 3, padding: 1)
        let x = MLXRandom.normal([2, 312, 448])
        let out = mlp(x)
        eval(out)
        #expect(out.shape == [2, 312, 448])
    }

    @Test func timestepEmbedderShape() {
        let embed = MATimestepEmbedder(dim: 448, frequencyEmbeddingSize: 256, maxPeriod: 10000)
        let t = MLXArray([0.5, 0.8]).asType(.float32)
        let out = embed(t)
        eval(out)
        #expect(out.shape == [2, 448])
    }

    @Test func ropeShape() {
        let rot = MARoPE.computeRotations(length: 312, dim: 64)
        eval(rot)
        // (1, length, dim/2, 2, 2)
        #expect(rot.shape == [1, 312, 32, 2, 2])
    }

    @Test func selfAttentionShape() {
        let attn = MASelfAttention(dim: 448, nheads: 7)
        let x = MLXRandom.normal([2, 312, 448])
        let (q, k, v) = attn.preAttention(x, rot: nil)
        eval(q, k, v)
        // (B, nheads, N, headDim)
        #expect(q.shape == [2, 7, 312, 64])
        #expect(k.shape == [2, 7, 312, 64])
        #expect(v.shape == [2, 7, 312, 64])
    }

    @Test func attentionOutputShape() {
        let q = MLXRandom.normal([2, 7, 312, 64])
        let k = MLXRandom.normal([2, 7, 312, 64])
        let v = MLXRandom.normal([2, 7, 312, 64])
        let out = maAttention(q: q, k: k, v: v)
        eval(out)
        #expect(out.shape == [2, 312, 448])
    }

    @Test func mmDitBlockConvShape() {
        let block = MAMMDitSingleBlock(dim: 448, nhead: 7, kernelSize: 3, padding: 1)
        let x = MLXRandom.normal([2, 312, 448])
        let cond = MLXRandom.normal([2, 1, 448])
        let out = block(x, cond: cond, rot: nil)
        eval(out)
        #expect(out.shape == [2, 312, 448])
    }

    @Test func mmDitBlockLinearShape() {
        let block = MAMMDitSingleBlock(dim: 448, nhead: 7, kernelSize: 1, padding: 0)
        let x = MLXRandom.normal([2, 77, 448])
        let cond = MLXRandom.normal([2, 1, 448])
        let out = block(x, cond: cond, rot: nil)
        eval(out)
        #expect(out.shape == [2, 77, 448])
    }

    @Test func jointBlockShape() {
        let block = MAJointBlock(dim: 448, nhead: 7)
        let latent = MLXRandom.normal([2, 312, 448])
        let textF = MLXRandom.normal([2, 77, 448])
        let globalC = MLXRandom.normal([2, 1, 448])
        let extendedC = globalC
        let (outLatent, outText) = block(
            latent: latent, textF: textF,
            globalC: globalC, extendedC: extendedC,
            latentRot: nil, textRot: nil
        )
        eval(outLatent, outText)
        #expect(outLatent.shape == [2, 312, 448])
        #expect(outText.shape == [2, 77, 448])
    }

    @Test func finalBlockShape() {
        let block = MAFinalBlock(dim: 448, outDim: 20)
        let latent = MLXRandom.normal([2, 312, 448])
        let cond = MLXRandom.normal([2, 1, 448])
        let out = block(latent, c: cond)
        eval(out)
        #expect(out.shape == [2, 312, 20])
    }

    @Test func fullModelForwardPass() {
        let config = MeanAudioConfig.small
        let model = MeanAudioFlowTransformer(config: config)

        let latent = MLXRandom.normal([1, 312, 20])
        let textF = MLXRandom.normal([1, 77, 1024])
        let textFC = MLXRandom.normal([1, 512])

        let conditions = model.preprocessConditions(textF: textF, textFC: textFC)
        #expect(conditions.textF.shape == [1, 77, 448])
        #expect(conditions.textFC.shape == [1, 448])

        let t = MLXArray([1.0]).asType(.float32)
        let r = MLXArray([0.0]).asType(.float32)
        let flow = model.predictFlow(latent: latent, t: t, r: r, conditions: conditions)
        eval(flow)
        #expect(flow.shape == [1, 312, 20])
    }

    @Test func samplerShape() {
        let config = MeanAudioConfig.small
        let model = MeanAudioFlowTransformer(config: config)

        let noise = MLXRandom.normal([1, 312, 20])
        let textF = MLXRandom.normal([1, 77, 1024])
        let textFC = MLXRandom.normal([1, 512])

        let conditions = model.preprocessConditions(textF: textF, textFC: textFC)
        let emptyConditions = model.getEmptyConditions(batchSize: 1)

        let sampler = MeanFlowSampler(steps: 1)
        let output = sampler.sample(
            model: model,
            noise: noise,
            conditions: conditions,
            emptyConditions: emptyConditions,
            cfgStrength: 4.5
        )
        eval(output)
        #expect(output.shape == [1, 312, 20])
    }

    // MARK: - VAE Tests

    @Test func mpConv1dShape() {
        let conv = MAMPConv1D(inChannels: 20, outChannels: 384, kernelSize: 3)
        let x = MLXRandom.normal([1, 20, 312])
        let out = conv(x)
        eval(out)
        #expect(out.shape == [1, 384, 312])
    }

    @Test func resnetBlock1dSameShape() {
        let block = MAResnetBlock1D(inDim: 384, outDim: 384)
        let x = MLXRandom.normal([1, 384, 312])
        let out = block(x)
        eval(out)
        #expect(out.shape == [1, 384, 312])
    }

    @Test func resnetBlock1dDiffShape() {
        let block = MAResnetBlock1D(inDim: 384, outDim: 768)
        let x = MLXRandom.normal([1, 384, 312])
        let out = block(x)
        eval(out)
        #expect(out.shape == [1, 768, 312])
    }

    @Test func attnBlock1dShape() {
        let block = MAAttnBlock1D(inChannels: 384)
        let x = MLXRandom.normal([1, 384, 32])
        let out = block(x)
        eval(out)
        #expect(out.shape == [1, 384, 32])
    }

    @Test func upsample1dShape() {
        let up = MAUpsample1D(inChannels: 384, withConv: true)
        let x = MLXRandom.normal([1, 384, 312])
        let out = up(x)
        eval(out)
        #expect(out.shape == [1, 384, 624])
    }

    @Test func vaeDecoderShape() {
        let decoder = MAVAEDecoder()
        let z = MLXRandom.normal([1, 20, 312])
        let mel = decoder(z)
        eval(mel)
        #expect(mel.dim(0) == 1)
        #expect(mel.dim(1) == 80)
        // Seq length depends on upsample (312 → 624 with one upsample layer)
        #expect(mel.dim(2) == 624)
    }

    @Test func vaeFullDecodeShape() {
        let vae = MeanAudioVAE()
        let z = MLXRandom.normal([1, 20, 312])
        let mel = vae.decode(z)
        eval(mel)
        #expect(mel.dim(0) == 1)
        #expect(mel.dim(1) == 80)
    }

    // MARK: - Pipeline Test

    @Test func pipelineConstruction() {
        let config = MeanAudioConfig.small
        let pipeline = MeanAudioPipeline(config: config)
        #expect(pipeline.parameterCount > 0)
    }
}
