#if canImport(MLX)
import Foundation
import MLX

/// 窓長が hop の整数倍の STFT（MLX、微分可能）
///
/// フレーム化は hop 単位のチャンクを k 個つなぐ形で行い、asStrided を使わない。
/// 入力は [B, N]（N は hop の倍数）。フレーム t はサンプル t·hop ..< t·hop + win（末尾は 0 埋め）
public final class MLXSTFT: @unchecked Sendable {
    public let fftSize: Int
    public let windowSize: Int
    public let hopSize: Int
    public let bins: Int
    let cosBasis: MLXArray   // [windowSize, bins]（窓を掛け込み済み）
    let sinBasis: MLXArray

    public init(fftSize: Int, windowSize: Int, hopSize: Int, window: [Float]) {
        precondition(windowSize % hopSize == 0, "windowSize must be a multiple of hopSize")
        precondition(window.count == windowSize)
        self.fftSize = fftSize
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.bins = fftSize / 2 + 1
        var c = [Float](repeating: 0.0, count: windowSize * bins)
        var s = [Float](repeating: 0.0, count: windowSize * bins)
        var n = 0
        while n < windowSize {
            var k = 0
            while k < bins {
                let theta = 2.0 * Float.pi * Float(k) * Float(n) / Float(fftSize)
                c[n * bins + k] = window[n] * cos(theta)
                s[n * bins + k] = -window[n] * sin(theta)
                k += 1
            }
            n += 1
        }
        self.cosBasis = MLXArray(c, [windowSize, bins])
        self.sinBasis = MLXArray(s, [windowSize, bins])
    }

    /// 周期 Hann 窓の STFT
    public convenience init(fftSize: Int, hopSize: Int) {
        self.init(fftSize: fftSize, windowSize: fftSize, hopSize: hopSize, window: SynthesisSTFT.window(size: fftSize))
    }

    /// pcm [B, N] → フレーム [B, T, windowSize]（T = N / hop）
    public func frames(_ pcm: MLXArray) -> MLXArray {
        let b = pcm.shape[0]
        let n = pcm.shape[1]
        let t = n / hopSize
        let k = windowSize / hopSize
        let chunks = pcm[0..., 0..<(t * hopSize)].reshaped([b, t, hopSize])
        let padded = concatenated([chunks, MLXArray.zeros([b, k - 1, hopSize])], axis: 1)
        var parts: [MLXArray] = []
        var i = 0
        while i < k {
            parts.append(padded[0..., i..<(i + t), 0...])
            i += 1
        }
        return concatenated(parts, axis: -1)
    }

    /// pcm [B, N] → 振幅 [B, T, bins]
    public func magnitude(_ pcm: MLXArray) -> MLXArray {
        let f = frames(pcm)
        let re = matmul(f, cosBasis)
        let im = matmul(f, sinBasis)
        return sqrt(re * re + im * im + 1e-7)
    }

    /// pcm [B, N] → パワー [B, T, bins]
    public func power(_ pcm: MLXArray) -> MLXArray {
        let f = frames(pcm)
        let re = matmul(f, cosBasis)
        let im = matmul(f, sinBasis)
        return re * re + im * im
    }
}

/// CPU の MelSpectrogramExtractor と同じ対数 Mel を MLX で計算する（微分可能）
public final class MLXMelAnalyzer: @unchecked Sendable {
    public let stft: MLXSTFT
    public let filterbank: MLXArray   // [bins, melChannels]
    public let melChannels: Int

    public init(extractor: MelSpectrogramExtractor = MelSpectrogramExtractor()) {
        self.stft = MLXSTFT(fftSize: extractor.fftSize, windowSize: extractor.frameSize, hopSize: extractor.hopSize, window: extractor.analysisWindow)
        self.melChannels = extractor.melChannels
        self.filterbank = MLXArray(extractor.filterbankWeights, [extractor.fftBins, extractor.melChannels])
    }

    /// pcm [B, N] → 対数 Mel [B, T, melChannels]
    public func logMel(_ pcm: MLXArray) -> MLXArray {
        let mel = matmul(stft.power(pcm), filterbank)
        return log(maximum(mel, MLXArray(MelSpectrogramExtractor.logMelFloor)))
    }
}

/// 複数解像度 STFT 損失（スペクトル収束 + 対数振幅 L1）
public final class MLXMultiResolutionSTFTLoss: @unchecked Sendable {
    public let stfts: [MLXSTFT]

    public init(resolutions: [(fft: Int, hop: Int)] = [(256, 64), (512, 128), (1024, 256)]) {
        self.stfts = resolutions.map { MLXSTFT(fftSize: $0.fft, hopSize: $0.hop) }
    }

    /// 返り値: [spectralConvergence, logMagnitudeL1]（解像度の平均）
    public func losses(predicted: MLXArray, target: MLXArray) -> (convergence: MLXArray, logMagnitude: MLXArray) {
        var sc = MLXArray(Float(0))
        var lm = MLXArray(Float(0))
        for stft in stfts {
            let p = stft.magnitude(predicted)
            let t = stft.magnitude(target)
            sc = sc + sqrt(sum(square(t - p))) / (sqrt(sum(square(t))) + 1e-6)
            lm = lm + mean(abs(log(t) - log(p)))
        }
        let n = Float(stfts.count)
        return (sc / n, lm / n)
    }
}
#endif
