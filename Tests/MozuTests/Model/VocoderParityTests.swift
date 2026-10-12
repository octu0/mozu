import XCTest
import Foundation
import MLX
@testable import Mozu

/// ボコーダの MLX 合成と CPU 合成の一致、および逆 STFT が STFT の逆になっていることを検証する
final class VocoderParityTests: XCTestCase {
    func testCPUMatchesMLX() {
        let config = VocoderDefaults.config(modelDim: 32, layers: 1)
        let model = MLXVocoder(config: config)
        let weights = model.exportWeights()
        let cpu = Vocoder(weights: weights)
        let frames = 12
        var mel: [[Float]] = []
        var t = 0
        while t < frames {
            var row: [Float] = []
            var c = 0
            while c < config.melChannels {
                row.append(-3.0 + 0.1 * Float((t * 7 + c * 3) % 11))
                c += 1
            }
            mel.append(row)
            t += 1
        }
        let flat = mel.flatMap { $0 }
        let mlxOut = model.synthesize(MLXArray(flat, [1, frames, config.melChannels]))
        eval(mlxOut)
        let mlx = mlxOut.asArray(Float.self)
        let cpuOut = cpu.synthesize(mel: mel)
        XCTAssertEqual(cpuOut.count, frames * config.hopSize)
        XCTAssertEqual(mlx.count, cpuOut.count)
        var i = 0
        var maxDiff: Float = 0
        while i < cpuOut.count {
            maxDiff = max(maxDiff, abs(cpuOut[i] - mlx[i]))
            i += 1
        }
        XCTAssertLessThan(maxDiff, 2e-3, "CPU と MLX の波形が一致しません")
    }

    /// 既知の波形を STFT し、その振幅・位相をヘッド出力として逆 STFT すると元の波形に戻る
    func testInverseSTFTReconstructsSignal() {
        let config = VocoderDefaults.config(modelDim: 32, layers: 1)
        let n = config.fftSize
        let hop = config.hopSize
        let frames = 20
        let total = frames * hop
        var signal = [Float](repeating: 0, count: total)
        var i = 0
        while i < total {
            signal[i] = 0.5 * sin(2 * Float.pi * 220 * Float(i) / 16000) + 0.2 * sin(2 * Float.pi * 1330 * Float(i) / 16000)
            i += 1
        }
        // 解析: 合成フレームと同じ位置・同じ周期 Hann 窓で STFT
        let window = SynthesisSTFT.window(size: n)
        let bins = config.bins
        var head = [Float](repeating: 0, count: frames * config.headDim)
        var t = 0
        while t < frames {
            let start = SynthesisSTFT.frameStart(frame: t, fftSize: n, hopSize: hop, analysisFrameSize: AudioConfig.frameSize)
            var k = 0
            while k < bins {
                var re: Float = 0
                var im: Float = 0
                var m = 0
                while m < n {
                    let idx = start + m
                    let x = (0 <= idx && idx < total) ? signal[idx] * window[m] : 0
                    let theta = 2 * Float.pi * Float(k) * Float(m) / Float(n)
                    re += x * cos(theta)
                    im -= x * sin(theta)
                    m += 1
                }
                let mag = sqrt(re * re + im * im)
                head[t * config.headDim + k] = log(max(mag, 1e-6))
                head[t * config.headDim + bins + k] = atan2(im, re)
                k += 1
            }
            t += 1
        }
        let basis = SynthesisSTFT.inverseBasis(fftSize: n)
        let out = Vocoder.inverseSTFT(head: head, frames: frames, config: config, cosBasis: basis.cos, sinBasis: basis.sin, window: window)
        // 端のフレームは隣の窓が無く再構成できないので中央部だけ比べる
        var err: Float = 0
        var ref: Float = 0
        i = n
        while i < total - n {
            err += (out[i] - signal[i]) * (out[i] - signal[i])
            ref += signal[i] * signal[i]
            i += 1
        }
        XCTAssertLessThan(err / max(ref, 1e-6), 1e-3, "逆 STFT が元の波形を再構成できていません")
    }
}
