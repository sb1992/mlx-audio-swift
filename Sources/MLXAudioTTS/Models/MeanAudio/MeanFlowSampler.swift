//
//  MeanFlowSampler.swift
//  MLXAudio
//
//  MeanFlow Euler ODE sampler for single-step (1-NFE) and multi-step inference.
//  Ported from MeanAudio Python: meanaudio/model/mean_flow.py
//

import Foundation
@preconcurrency import MLX

struct MeanFlowSampler {
    let steps: Int

    init(steps: Int = 1) {
        self.steps = steps
    }

    func sample(
        model: MeanAudioFlowTransformer,
        noise: MLXArray,
        conditions: MAPreprocessedConditions,
        emptyConditions: MAPreprocessedConditions,
        cfgStrength: Float
    ) -> MLXArray {
        var x = noise

        // Steps from t=1 → t=0. linspace(1, 0, steps+1) gives the time grid.
        let timeSteps = MLX.linspace(Float32(1.0), Float32(0.0), count: steps + 1)
        eval(timeSteps)

        for i in 0 ..< steps {
            let t = timeSteps[i]
            let nextT = timeSteps[i + 1]

            let flow = model.odeWrapper(
                t: t, r: nextT,
                latent: x,
                conditions: conditions,
                emptyConditions: emptyConditions,
                cfgStrength: cfgStrength
            )

            let dt = t - nextT
            x = x - dt * flow
            eval(x)
        }

        return x
    }
}
