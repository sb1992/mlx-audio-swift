# MeanAudio

Text-to-audio generation using a Flux-style MMDiT (Multimodal Diffusion Transformer) with MeanFlow single-step ODE inference. Generates sound effects and audio from text descriptions. 120M parameters, ~10 seconds of 16kHz audio per generation.

## Architecture

- **Flow Transformer**: Joint attention MMDiT with dual timestep embeddings (t + r), AdaLN modulation, gated SwiGLU MLP. 4 joint blocks + 8 fused audio-only blocks.
- **VAE Decoder**: EDM2-style magnitude-preserving VAE. Converts latent (20-dim) to 80-bin mel spectrogram.
- **BigVGAN Vocoder**: Converts mel spectrogram to 16kHz waveform. (Uses existing MLXAudioCodecs BigVGAN.)
- **Text Encoding**: FLAN-T5-Large (1024-dim) + LAION CLAP (512-dim) via Python sidecar.

## Usage

MeanAudio requires a Python sidecar process for text encoding (T5 + CLAP are large models best run via PyTorch).

### 1. Start the text encoder sidecar

```bash
pip install torch transformers laion-clap numpy
python Tools/MeanAudio/text_encoder_sidecar.py serve --port 8765
```

### 2. Generate audio from Swift

```swift
import MLXAudioTTS

let model = try await MeanAudioModel.fromPretrained("sb1992/meanaudio-small-mlx")
let audio = try await model.generate(
    text: "A dog barking in a park with birds chirping",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(temperature: 0.0)
)
// audio is a 1D MLXArray of float32 samples at 16kHz
```

### Direct API (with pre-computed features)

```swift
let features = try await MeanAudioTextEncoderClient().encode(text: "Thunder rolling")
let waveform = try model.generateFromFeatures(
    textFeatures: features.textFeatures,
    textFeaturesC: features.textFeaturesC,
    options: MeanAudioGenerateOptions(cfgStrength: 4.5, steps: 1, seed: 42)
)
```

## Weight Conversion

Convert PyTorch checkpoints to MLX safetensors:

```bash
# Convert all components at once
python Tools/MeanAudio/convert_meanaudio_weights.py bundle \
    --flow path/to/ema_ckpt.pth \
    --vae path/to/vae.pth \
    --bigvgan path/to/bigvgan.pth \
    --output path/to/mlx-weights/

# Or convert individually
python Tools/MeanAudio/convert_meanaudio_weights.py flow --input ema_ckpt.pth --output weights/
python Tools/MeanAudio/convert_meanaudio_weights.py vae --input vae.pth --output weights/
python Tools/MeanAudio/convert_meanaudio_weights.py bigvgan --input bigvgan.pth --output weights/
```

## Model Files (HuggingFace repo layout)

```
config.json              # MeanAudioConfig
model.safetensors        # Flow transformer weights
vae.safetensors          # VAE decoder weights
bigvgan.safetensors      # BigVGAN vocoder weights
bigvgan_config.json      # BigVGAN config
```

## License

MIT (upstream MeanAudio is MIT-licensed).
