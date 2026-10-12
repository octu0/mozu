#if canImport(MLX)
import Foundation
import MLX
import MLXNN

/// 逆 STFT（MLX）。学習しない定数なので Module の外に置く
///
/// なぜ Module のプロパティにしないか: Module は MLXArray のプロパティをすべてパラメータとして集めるため、
/// 基底行列や窓を Module に持たせると最適化器がそれらを更新してしまい、CPU 推論（固定の基底）と食い違う。
public final class MLXInverseSTFT: @unchecked Sendable {
    public let config: VocoderConfig
    let cosBasis: MLXArray   // [bins, fftSize]
    let sinBasis: MLXArray
    let window: MLXArray     // [fftSize]

    public init(config: VocoderConfig) {
        self.config = config
        let basis = SynthesisSTFT.inverseBasis(fftSize: config.fftSize)
        self.cosBasis = MLXArray(basis.cos, [config.bins, config.fftSize])
        self.sinBasis = MLXArray(basis.sin, [config.bins, config.fftSize])
        self.window = MLXArray(SynthesisSTFT.window(size: config.fftSize))
    }

    /// ヘッド出力 [B, T, 2·bins]（対数振幅, 位相）→ 波形 [B, T·hop]（CPU の Vocoder.inverseSTFT と同じ式）
    public func waveform(head: MLXArray) -> MLXArray {
        let b = head.shape[0]
        let t = head.shape[1]
        let bins = config.bins
        let n = config.fftSize
        let hop = config.hopSize
        let k = n / hop
        let logMag = minimum(head[0..., 0..., 0..<bins], MLXArray(VocoderConstants.maxLogMagnitude))
        let mag = exp(logMag)
        let phase = head[0..., 0..., bins...]
        let re = mag * cos(phase)
        let im = mag * sin(phase)
        let frames = (matmul(re, cosBasis) + matmul(im, sinBasis)) * window     // [B, T, n]
        let chunks = frames.reshaped([b, t, k, hop])
        var ola = MLXArray.zeros([b, t + k - 1, hop])
        var i = 0
        while i < k {
            let part = chunks[0..., 0..., i, 0...]                               // [B, T, hop]
            let padded = concatenated([MLXArray.zeros([b, i, hop]), part, MLXArray.zeros([b, k - 1 - i, hop])], axis: 1)
            ola = ola + padded
            i += 1
        }
        let flat = ola.reshaped([b, (t + k - 1) * hop])
        // 合成フレーム 0 の先頭は実サンプル frameStart(0)（負）なので、実サンプル 0 は OLA 配列の −frameStart(0) 番目
        let offset = -SynthesisSTFT.frameStart(frame: 0, fftSize: n, hopSize: hop, analysisFrameSize: AudioConfig.frameSize)
        let total = t * hop
        let cropped = flat[0..., offset..<(offset + total)]
        let norm = MLXArray(SynthesisSTFT.windowSquareSum(frames: t, fftSize: n, hopSize: hop, analysisFrameSize: AudioConfig.frameSize))
        return cropped / norm
    }
}

/// ボコーダ（MLX、学習用）
public final class MLXVocoder: Module, @unchecked Sendable {
    public let config: VocoderConfig
    public var inputProjection: MLXArray
    public var inputBias: MLXArray
    public var stack: MLXHawkStack
    public var outputNorm: MLXArray
    public var headProjection: MLXArray
    public var headBias: MLXArray
    public var positionEmbedding: MLXArray
    public let istft: MLXInverseSTFT

    public init(config: VocoderConfig) {
        self.config = config
        let d = config.modelDim
        let inBound = sqrt(3.0 / Float(config.melChannels))
        self.inputProjection = MLXRandom.uniform(low: -inBound, high: inBound, [config.melChannels, d])
        // 無音フレームでも残差が 0 にならないよう入力バイアスを持つ（RMSNorm の逆伝播が 1/√ε 倍に膨らむのを防ぐ）
        let biasBound = 0.1 * sqrt(Float(3.0))
        self.inputBias = MLXRandom.uniform(low: -biasBound, high: biasBound, [d])
        self.stack = MLXHawkStack(config: config.stack)
        self.outputNorm = MLXArray.ones([d])
        let outBound = sqrt(3.0 / Float(d))
        self.headProjection = MLXRandom.uniform(low: -outBound, high: outBound, [d, config.headDim])
        self.headBias = MLXArray.zeros([config.headDim])
        self.positionEmbedding = MLXRandom.uniform(low: -0.1, high: 0.1, [config.positionPeriod, d])
        self.istft = MLXInverseSTFT(config: config)
        super.init()
    }

    public convenience init(weights: VocoderWeights) {
        self.init(config: weights.config)
        load(weights)
    }

    public func load(_ w: VocoderWeights) {
        inputProjection = MLXArray(w.inputProjection.values, w.inputProjection.shape)
        inputBias = MLXArray(w.inputBias.values, w.inputBias.shape)
        var l = 0
        while l < w.stack.layers.count {
            stack.layers[l].load(w.stack.layers[l])
            l += 1
        }
        outputNorm = MLXArray(w.outputNorm.values, w.outputNorm.shape)
        headProjection = MLXArray(w.headProjection.values, w.headProjection.shape)
        headBias = MLXArray(w.headBias.values, w.headBias.shape)
        positionEmbedding = MLXArray(w.positionEmbedding.values, w.positionEmbedding.shape)
    }

    public func exportWeights() -> VocoderWeights {
        return VocoderWeights(
            config: config,
            inputProjection: tensor(inputProjection),
            inputBias: tensor(inputBias),
            stack: stack.exportWeights(),
            outputNorm: tensor(outputNorm),
            headProjection: tensor(headProjection),
            headBias: tensor(headBias),
            positionEmbedding: tensor(positionEmbedding)
        )
    }

    /// mel [B, T, melChannels] → ヘッド出力 [B, T, headDim]
    public func head(_ mel: MLXArray, frameMask: MLXArray? = nil) -> MLXArray {
        let frames = mel.shape[1]
        let period = config.positionPeriod
        let positions = MLXArray((0..<frames).map { Int32($0 % period) })
        let x = matmul(mel, inputProjection) + inputBias + take(positionEmbedding, positions, axis: 0)
        let h = hawkRMSNorm(stack(x, frameMask: frameMask), outputNorm)
        return matmul(h, headProjection) + headBias
    }

    /// mel → 波形
    public func synthesize(_ mel: MLXArray, frameMask: MLXArray? = nil) -> MLXArray {
        return istft.waveform(head: head(mel, frameMask: frameMask))
    }
}

/// ボコーダの既定構成
public enum VocoderDefaults {
    public static func config(modelDim: Int = 256, layers: Int = 6) -> VocoderConfig {
        let stack = HawkStackConfig(modelDim: modelDim, rnnDim: modelDim, mlpDim: modelDim * 3, heads: 4, convLeft: 2, convRight: 2, numLayers: layers, bidirectional: true)
        return VocoderConfig(melChannels: AudioConfig.melChannels, modelDim: modelDim, stack: stack, fftSize: 640, hopSize: AudioConfig.hopSize)
    }
}
#endif
