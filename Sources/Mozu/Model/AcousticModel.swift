import Foundation

/// 音響モデルの CPU 推論
public final class AcousticModel: @unchecked Sendable {
    public let weights: AcousticWeights
    private let encoder: HawkStack
    private let decoder: HawkStack

    public struct Output: Sendable {
        /// [frames][melChannels] 対数 Mel
        public let mel: [[Float]]
        /// 音素ごとのフレーム数
        public let durations: [Int]
        /// 音素ごとの log2(F0/220)（正規化を戻した値）
        public let pitch: [Float]
        public let energy: [Float]
    }

    public init(weights: AcousticWeights) {
        self.weights = weights
        self.encoder = HawkStack(weights: weights.encoder)
        self.decoder = HawkStack(weights: weights.decoder)
    }

    public var config: AcousticConfig {
        return weights.config
    }

    /// phonemeIds: 音素 ID 列（文頭・文末の無音を含む）、features: [phoneCount][featureCount]、
    /// labelIndices: config.labelKinds の順のラベル index（0 = 未指定）
    /// speed: 話速倍率、pitchShift: オクターブ単位のピッチ移動、durations: 与えると継続時間予測を使わない
    public func synthesize(
        phonemeIds: [Int],
        features: [[Float]],
        labelIndices: [Int],
        speed: Float = 1.0,
        pitchShift: Float = 0.0,
        durations: [Int]? = nil,
        maxLeadSilenceFrames: Int = 8,
        maxTrailSilenceFrames: Int = 12
    ) -> Output {
        let cfg = weights.config
        let d = cfg.modelDim
        let phoneCount = phonemeIds.count
        if phoneCount == 0 {
            return Output(mel: [], durations: [], pitch: [], energy: [])
        }
        precondition(features.count == phoneCount)
        precondition(labelIndices.count == cfg.labelKinds.count)

        // 1. 音素埋め込み + 特徴射影 + ラベル埋め込み
        var x = [Float](repeating: 0.0, count: phoneCount * d)
        var style = [Float](repeating: 0.0, count: d)
        var k = 0
        while k < cfg.labelKinds.count {
            let table = weights.labelEmbeddings[k]
            let rows = table.shape[0]
            let idx = min(max(0, labelIndices[k]), rows - 1)
            var c = 0
            while c < d {
                style[c] += table.values[idx * d + c]
                c += 1
            }
            k += 1
        }
        var p = 0
        while p < phoneCount {
            let id = min(max(0, phonemeIds[p]), cfg.phonemeVocabSize - 1)
            let base = p * d
            var c = 0
            while c < d {
                x[base + c] = weights.phonemeEmbedding.values[id * d + c] + style[c]
                c += 1
            }
            var f = 0
            while f < cfg.featureCount {
                let v = f < features[p].count ? features[p][f] : 0.0
                if v != 0.0 {
                    c = 0
                    while c < d {
                        x[base + c] += v * weights.featureProjection.values[f * d + c]
                        c += 1
                    }
                }
                f += 1
            }
            p += 1
        }

        // 2. エンコーダ
        var h = encoder.forward(x, frames: phoneCount)
        var normed = [Float](repeating: 0.0, count: phoneCount * d)
        HawkStack.rmsNorm(h, weights.encoderNorm.values, &normed, frames: phoneCount, width: d)

        // 3. 継続時間・ピッチ・エネルギー
        let logDur = Self.head(weights.durationHead, normed, frames: phoneCount, inDim: d, hidden: cfg.headHiddenDim)
        var pitchZ = Self.head(weights.pitchHead, normed, frames: phoneCount, inDim: d, hidden: cfg.headHiddenDim)
        let energyZ = Self.head(weights.energyHead, normed, frames: phoneCount, inDim: d, hidden: cfg.headHiddenDim)
        // 無声化母音など F0 教師の無い音素は損失でマスクされ予測値が定まらないので、学習時の分布（±3σ）に収める
        var i = 0
        while i < pitchZ.count {
            pitchZ[i] = min(3.0, max(-3.0, pitchZ[i]))
            if pitchShift != 0.0 && 0.0 < cfg.pitchStd {
                pitchZ[i] += pitchShift / cfg.pitchStd
            }
            i += 1
        }
        var frameCounts: [Int] = []
        switch durations {
        case .some(let given) where given.count == phoneCount:
            frameCounts = given.map { max(1, $0) }
        case _:
            let safeSpeed = max(0.25, min(4.0, speed))
            p = 0
            while p < phoneCount {
                let frames = Int((expf(logDur[p]) / safeSpeed).rounded())
                frameCounts.append(max(1, frames))
                p += 1
            }
            frameCounts[0] = min(frameCounts[0], maxLeadSilenceFrames)
            frameCounts[phoneCount - 1] = min(frameCounts[phoneCount - 1], maxTrailSilenceFrames)
        }

        // 4. ピッチ・エネルギー埋め込みを足して長さ調整
        p = 0
        while p < phoneCount {
            let base = p * d
            var c = 0
            while c < d {
                normed[base + c] += pitchZ[p] * weights.pitchProjection.w.values[c] + weights.pitchProjection.b.values[c]
                    + energyZ[p] * weights.energyProjection.w.values[c] + weights.energyProjection.b.values[c]
                c += 1
            }
            p += 1
        }
        let totalFrames = frameCounts.reduce(0, +)
        var frames = [Float](repeating: 0.0, count: totalFrames * d)
        var t = 0
        p = 0
        while p < phoneCount {
            let dur = frameCounts[p]
            var j = 0
            while j < dur {
                let pos = (Float(j) + 0.5) / Float(dur)
                let src = p * d
                let dst = t * d
                var c = 0
                while c < d {
                    frames[dst + c] = normed[src + c] + pos * weights.positionProjection.w.values[c] + weights.positionProjection.b.values[c]
                    c += 1
                }
                t += 1
                j += 1
            }
            p += 1
        }

        // 5. デコーダ → Mel
        h = decoder.forward(frames, frames: totalFrames)
        var decNormed = [Float](repeating: 0.0, count: totalFrames * d)
        HawkStack.rmsNorm(h, weights.decoderNorm.values, &decNormed, frames: totalFrames, width: d)
        let melDim = cfg.melChannels
        var melFlat = [Float](repeating: 0.0, count: totalFrames * melDim)
        t = 0
        while t < totalFrames {
            var c = 0
            while c < melDim {
                melFlat[t * melDim + c] = weights.melBias.values[c]
                c += 1
            }
            t += 1
        }
        Gemm.multiply(decNormed, weights.melProjection.values, &melFlat, m: totalFrames, n: melDim, k: d, beta: 1.0)
        var mel: [[Float]] = []
        mel.reserveCapacity(totalFrames)
        t = 0
        while t < totalFrames {
            mel.append(Array(melFlat[(t * melDim)..<((t + 1) * melDim)]))
            t += 1
        }
        let pitch = pitchZ.map { $0 * cfg.pitchStd + cfg.pitchMean }
        let energy = energyZ.map { $0 * cfg.energyStd + cfg.energyMean }
        return Output(mel: mel, durations: frameCounts, pitch: pitch, energy: energy)
    }

    /// Linear → GeLU → Linear（スカラー出力）
    static func head(_ w: HeadWeights, _ x: [Float], frames: Int, inDim: Int, hidden: Int) -> [Float] {
        var h = [Float](repeating: 0.0, count: frames * hidden)
        var t = 0
        while t < frames {
            var c = 0
            while c < hidden {
                h[t * hidden + c] = w.b1.values[c]
                c += 1
            }
            t += 1
        }
        Gemm.multiply(x, w.w1.values, &h, m: frames, n: hidden, k: inDim, beta: 1.0)
        HawkStack.geluInPlace(&h)
        var out = [Float](repeating: w.b2.values[0], count: frames)
        Gemm.multiply(h, w.w2.values, &out, m: frames, n: 1, k: hidden, beta: 1.0)
        return out
    }
}
