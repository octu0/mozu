import Foundation

/// ボコーダの CPU 推論
public final class Vocoder: @unchecked Sendable {
    public let weights: VocoderWeights
    private let stack: HawkStack
    private let cosBasis: [Float]
    private let sinBasis: [Float]
    private let window: [Float]

    public init(weights: VocoderWeights) {
        self.weights = weights
        self.stack = HawkStack(weights: weights.stack)
        let basis = SynthesisSTFT.inverseBasis(fftSize: weights.config.fftSize)
        self.cosBasis = basis.cos
        self.sinBasis = basis.sin
        self.window = SynthesisSTFT.window(size: weights.config.fftSize)
    }

    public var config: VocoderConfig {
        return weights.config
    }

    /// mel: [frames][melChannels] → PCM（frames · hop サンプル）
    public func synthesize(mel: [[Float]]) -> [Float] {
        if mel.isEmpty {
            return []
        }
        return Self.inverseSTFT(head: head(mel: mel), frames: mel.count, config: weights.config, cosBasis: cosBasis, sinBasis: sinBasis, window: window)
    }

    /// mel: [frames][melChannels] → ヘッド出力 [frames, 2·bins]（対数振幅, 位相）
    public func head(mel: [[Float]]) -> [Float] {
        let cfg = weights.config
        let frames = mel.count
        let d = cfg.modelDim
        let melDim = cfg.melChannels
        var melFlat = [Float](repeating: 0.0, count: frames * melDim)
        var t = 0
        while t < frames {
            var c = 0
            while c < melDim {
                melFlat[t * melDim + c] = c < mel[t].count ? mel[t][c] : 0.0
                c += 1
            }
            t += 1
        }
        var x = [Float](repeating: 0.0, count: frames * d)
        t = 0
        while t < frames {
            var c = 0
            while c < d {
                x[t * d + c] = weights.inputBias.values[c]
                c += 1
            }
            t += 1
        }
        Gemm.multiply(melFlat, weights.inputProjection.values, &x, m: frames, n: d, k: melDim, beta: 1.0)
        let period = cfg.positionPeriod
        t = 0
        while t < frames {
            let row = (t % period) * d
            var c = 0
            while c < d {
                x[t * d + c] += weights.positionEmbedding.values[row + c]
                c += 1
            }
            t += 1
        }
        let h = stack.forward(x, frames: frames)
        var normed = [Float](repeating: 0.0, count: frames * d)
        HawkStack.rmsNorm(h, weights.outputNorm.values, &normed, frames: frames, width: d)
        let headDim = cfg.headDim
        var head = [Float](repeating: 0.0, count: frames * headDim)
        t = 0
        while t < frames {
            var c = 0
            while c < headDim {
                head[t * headDim + c] = weights.headBias.values[c]
                c += 1
            }
            t += 1
        }
        Gemm.multiply(normed, weights.headProjection.values, &head, m: frames, n: headDim, k: d, beta: 1.0)
        return head
    }

    /// ヘッド出力 [frames, 2·bins]（対数振幅, 位相）→ 波形
    static func inverseSTFT(head: [Float], frames: Int, config: VocoderConfig, cosBasis: [Float], sinBasis: [Float], window: [Float]) -> [Float] {
        let bins = config.bins
        let n = config.fftSize
        var re = [Float](repeating: 0.0, count: frames * bins)
        var im = [Float](repeating: 0.0, count: frames * bins)
        var t = 0
        while t < frames {
            var k = 0
            while k < bins {
                let logMag = min(head[t * config.headDim + k], VocoderConstants.maxLogMagnitude)
                let mag = expf(logMag)
                let phase = head[t * config.headDim + bins + k]
                re[t * bins + k] = mag * cos(phase)
                im[t * bins + k] = mag * sin(phase)
                k += 1
            }
            t += 1
        }
        var frameBuf = [Float](repeating: 0.0, count: frames * n)
        Gemm.multiply(re, cosBasis, &frameBuf, m: frames, n: n, k: bins, beta: 0.0)
        Gemm.multiply(im, sinBasis, &frameBuf, m: frames, n: n, k: bins, beta: 1.0)
        let total = frames * config.hopSize
        var out = [Float](repeating: 0.0, count: total)
        t = 0
        while t < frames {
            let start = SynthesisSTFT.frameStart(frame: t, fftSize: n, hopSize: config.hopSize, analysisFrameSize: AudioConfig.frameSize)
            var i = 0
            while i < n {
                let idx = start + i
                if 0 <= idx && idx < total {
                    out[idx] += frameBuf[t * n + i] * window[i]
                }
                i += 1
            }
            t += 1
        }
        let norm = SynthesisSTFT.windowSquareSum(frames: frames, fftSize: n, hopSize: config.hopSize, analysisFrameSize: AudioConfig.frameSize)
        var i = 0
        while i < total {
            out[i] /= norm[i]
            i += 1
        }
        return out
    }
}

public enum VocoderConstants {
    /// 対数振幅の上限（学習初期の発散を防ぐ。e^7 ≈ 1100）
    public static let maxLogMagnitude: Float = 7.0
}
