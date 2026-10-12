#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// 音響モデルの損失の内訳
public struct AcousticLosses: Sendable {
    public var total: Float = 0
    public var mel: Float = 0
    public var duration: Float = 0
    public var pitch: Float = 0
    public var energy: Float = 0

    public static func + (a: AcousticLosses, b: AcousticLosses) -> AcousticLosses {
        return AcousticLosses(total: a.total + b.total, mel: a.mel + b.mel, duration: a.duration + b.duration, pitch: a.pitch + b.pitch, energy: a.energy + b.energy)
    }

    public func scaled(_ s: Float) -> AcousticLosses {
        return AcousticLosses(total: total * s, mel: mel * s, duration: duration * s, pitch: pitch * s, energy: energy * s)
    }

    public var description: String {
        return String(format: "total %.4f (mel %.4f, dur %.4f, pitch %.4f, energy %.4f)", total, mel, duration, pitch, energy)
    }

    /// 採択に使う値（Mel とピッチ。継続時間は過学習しやすく、採択基準に入れると Mel の学習を捨ててしまう）
    public var selection: Float {
        return mel + pitch
    }
}

/// 音響モデルの学習
public final class MLXAcousticTrainer: @unchecked Sendable {
    public let model: MLXAcousticModel
    public let optimizer: Adam
    public var melWeight: Float = 1.0
    public var durationWeight: Float = 0.5
    public var pitchWeight: Float = 0.5
    public var energyWeight: Float = 0.5
    public var gradientClip: Float = 5.0
    /// 学習時にラベルを未指定 (0) に置き換える確率。ラベル無しの合成でも破綻しないようにする
    public var labelDropout: Float = 0.1
    public private(set) var lastGradientNorm: Float = 0

    public init(model: MLXAcousticModel, learningRate: Float) {
        self.model = model
        self.optimizer = Adam(learningRate: learningRate)
    }

    public var learningRate: Float {
        get { optimizer.learningRate }
        set { optimizer.learningRate = newValue }
    }

    /// サンプル列をバッチにまとめる（詰め物とマスク）
    /// `mels` を与えると各サンプルの Mel の代わりに使う（遅延読み込み）
    public func makeBatch(_ samples: [UtteranceSample], mels: [[[Float]]]? = nil, dropLabels: Bool, rng: inout SystemRandomNumberGenerator) -> MLXAcousticModel.Batch {
        let cfg = model.config
        let b = samples.count
        let p = samples.map { $0.phoneCount }.max() ?? 1
        let t = samples.map { $0.frameCount }.max() ?? 1
        let f = cfg.featureCount
        let melDim = cfg.melChannels
        let k = cfg.labelKinds.count
        var ids = [Int32](repeating: 0, count: b * p)
        var feats = [Float](repeating: 0, count: b * p * f)
        var labels = [Int32](repeating: 0, count: b * k)
        var phoneMask = [Float](repeating: 0, count: b * p)
        var voicedMask = [Float](repeating: 0, count: b * p)
        var logDur = [Float](repeating: 0, count: b * p)
        var pitch = [Float](repeating: 0, count: b * p)
        var energy = [Float](repeating: 0, count: b * p)
        var frameIndex = [Int32](repeating: 0, count: b * t)
        var position = [Float](repeating: 0, count: b * t)
        var frameMask = [Float](repeating: 0, count: b * t)
        var mel = [Float](repeating: 0, count: b * t * melDim)
        var i = 0
        while i < b {
            let s = samples[i]
            var j = 0
            while j < s.phoneCount {
                let row = i * p + j
                ids[row] = s.phonemeIds[j]
                var c = 0
                while c < f {
                    feats[row * f + c] = c < s.features[j].count ? s.features[j][c] : 0
                    c += 1
                }
                phoneMask[row] = 1
                voicedMask[row] = s.voiced[j] ? 1 : 0
                logDur[row] = log(Float(max(1, s.durations[j])))
                pitch[row] = s.voiced[j] ? (s.pitch[j] - cfg.pitchMean) / cfg.pitchStd : 0
                energy[row] = (s.energy[j] - cfg.energyMean) / cfg.energyStd
                j += 1
            }
            var kk = 0
            while kk < k {
                var idx = kk < s.labelIndices.count ? s.labelIndices[kk] : 0
                if dropLabels && Float.random(in: 0..<1, using: &rng) < labelDropout {
                    idx = 0
                }
                labels[i * k + kk] = Int32(idx)
                kk += 1
            }
            var frame = 0
            j = 0
            while j < s.phoneCount {
                let dur = s.durations[j]
                var q = 0
                while q < dur && frame < t {
                    frameIndex[i * t + frame] = Int32(i * p + j)
                    position[i * t + frame] = (Float(q) + 0.5) / Float(dur)
                    frameMask[i * t + frame] = 1
                    frame += 1
                    q += 1
                }
                j += 1
            }
            // 詰め物のフレームは自分のバッチの先頭音素（無音）を指すようにして、index が範囲外にならないようにする
            while frame < t {
                frameIndex[i * t + frame] = Int32(i * p)
                frame += 1
            }
            let source = mels?[i] ?? s.mel
            var tt = 0
            while tt < min(s.frameCount, source.count) {
                let base = (i * t + tt) * melDim
                var c = 0
                while c < melDim {
                    mel[base + c] = source[tt][c]
                    c += 1
                }
                tt += 1
            }
            i += 1
        }
        return MLXAcousticModel.Batch(
            phonemeIds: MLXArray(ids, [b, p]),
            features: MLXArray(feats, [b, p, f]),
            labelIds: MLXArray(labels, [b, k]),
            phoneMask: MLXArray(phoneMask, [b, p, 1]),
            voicedMask: MLXArray(voicedMask, [b, p, 1]),
            logDurations: MLXArray(logDur, [b, p, 1]),
            pitch: MLXArray(pitch, [b, p, 1]),
            energy: MLXArray(energy, [b, p, 1]),
            frameIndex: MLXArray(frameIndex),
            position: MLXArray(position, [b, t, 1]),
            frameMask: MLXArray(frameMask, [b, t, 1]),
            mel: MLXArray(mel, [b, t, melDim]),
            batchSize: b,
            phoneCount: p,
            frameCount: t
        )
    }

    /// マスク付き損失 [total, mel, dur, pitch, energy]
    func losses(_ m: MLXAcousticModel, _ batch: MLXAcousticModel.Batch) -> [MLXArray] {
        let pred = m.forward(batch)
        let frameCount = maximum(sum(batch.frameMask), MLXArray(Float(1)))
        let phoneCount = maximum(sum(batch.phoneMask), MLXArray(Float(1)))
        let voicedCount = maximum(sum(batch.voicedMask), MLXArray(Float(1)))
        let melLoss = sum(abs(pred.mel - batch.mel) * batch.frameMask) / (frameCount * Float(m.config.melChannels))
        let durLoss = sum(square(pred.logDurations - batch.logDurations) * batch.phoneMask) / phoneCount
        // 無声音素の目標は 0（平均）にしてあり、ここも損失に含める。推論時に無声音素へ定まらない値が出て
        // デコーダの条件を乱すことを防ぐため（有声音素だけで学習すると無声音素の予測が自由になる）
        let pitchLoss = (sum(square(pred.pitch - batch.pitch) * batch.voicedMask) + 0.2 * sum(square(pred.pitch - batch.pitch) * (batch.phoneMask - batch.voicedMask))) / voicedCount
        let energyLoss = sum(square(pred.energy - batch.energy) * batch.phoneMask) / phoneCount
        let total = melLoss * melWeight + durLoss * durationWeight + pitchLoss * pitchWeight + energyLoss * energyWeight
        return [total, melLoss, durLoss, pitchLoss, energyLoss]
    }

    /// 1 バッチの更新
    public func trainStep(_ batch: MLXAcousticModel.Batch) -> AcousticLosses {
        let lossFn = { (m: MLXAcousticModel, _: [MLXArray]) -> [MLXArray] in
            return self.losses(m, batch)
        }
        let valueAndGradient = valueAndGrad(model: model, lossFn)
        let (values, grads) = valueAndGradient(model, [])
        let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: gradientClip)
        optimizer.update(model: model, gradients: clipped)
        // なぜここで optimizer も評価するか: Adam の step 等が未評価のまま連鎖すると Metal の資源上限で落ちる
        eval(model, optimizer, values, norm)
        lastGradientNorm = norm.item(Float.self)
        return Self.toLosses(values)
    }

    /// 勾配無しの評価
    public func evaluate(_ batch: MLXAcousticModel.Batch) -> AcousticLosses {
        let values = losses(model, batch)
        eval(values)
        return Self.toLosses(values)
    }

    static func toLosses(_ v: [MLXArray]) -> AcousticLosses {
        return AcousticLosses(total: v[0].item(Float.self), mel: v[1].item(Float.self), duration: v[2].item(Float.self), pitch: v[3].item(Float.self), energy: v[4].item(Float.self))
    }
}

/// 長さの近い発話をまとめるバッチ分割
public enum BatchPlanner {
    /// フレーム数でソートし、`batchSize` 件または詰め物込みの総フレーム数が `maxFrames` を超えない範囲でまとめ、
    /// バッチの順序をシャッフルする。長い発話だけのバッチは自動的に小さくなり、メモリの山を抑えられる
    public static func plan(sampleFrames: [Int], batchSize: Int, maxFrames: Int = Int.max, rng: inout SystemRandomNumberGenerator) -> [[Int]] {
        let order = sampleFrames.indices.sorted { sampleFrames[$0] < sampleFrames[$1] }
        var batches: [[Int]] = []
        var current: [Int] = []
        var longest = 0
        for idx in order {
            let frames = sampleFrames[idx]
            let nextLongest = max(longest, frames)
            if current.isEmpty != true && (current.count >= max(1, batchSize) || nextLongest * (current.count + 1) > maxFrames) {
                batches.append(current)
                current = []
                longest = 0
            }
            current.append(idx)
            longest = max(longest, frames)
        }
        if current.isEmpty != true {
            batches.append(current)
        }
        batches.shuffle(using: &rng)
        return batches
    }
}
#endif
