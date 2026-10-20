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
    /// 潜在変数の KL（free bits 適用前、次元の合計。style が無ければ 0）
    public var kl: Float = 0

    public static func + (a: AcousticLosses, b: AcousticLosses) -> AcousticLosses {
        return AcousticLosses(total: a.total + b.total, mel: a.mel + b.mel, duration: a.duration + b.duration, pitch: a.pitch + b.pitch, energy: a.energy + b.energy, kl: a.kl + b.kl)
    }

    public func scaled(_ s: Float) -> AcousticLosses {
        return AcousticLosses(total: total * s, mel: mel * s, duration: duration * s, pitch: pitch * s, energy: energy * s, kl: kl * s)
    }

    public var description: String {
        let base = String(format: "total %.4f (mel %.4f, dur %.4f, pitch %.4f, energy %.4f", total, mel, duration, pitch, energy)
        return base + (kl == 0 ? ")" : String(format: ", kl %.3f)", kl))
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
    /// 種類ごとの置き換え確率（labelDropout より優先）
    ///
    /// なぜ種類ごとに変えるか: 話者のように他のラベル（口調など）と完全に結び付いた種類は、それだけで話し方を説明できてしまい、
    /// 他の種類の埋め込みが育たない。その種類を高い確率で伏せると、伏せた間は他のラベルで説明するしかなくなる。
    public var labelDropoutByKind: [String: Float] = [:]
    public private(set) var lastGradientNorm: Float = 0
    /// 潜在変数の KL の重み（学習ループが徐々に上げる）
    public var klWeight: Float = 0
    /// 潜在変数の次元ごとの最低情報量（nats）。これ未満の KL には罰を与えず、次元が使われなくなるのを防ぐ
    public var freeBits: Float = 0.1
    /// 学習時に潜在変数を 0 に伏せる確率。参照なし（z = 0）の合成を学習済みの状態にするため
    public var styleDropout: Float = 0.2
    /// 直近のバッチの次元ごとの KL（ログ用）
    public private(set) var lastKLPerDim: [Float] = []

    /// 潜在変数の扱い: train は分布から引いて一部を伏せる、posterior は平均を使う、prior は 0（参照なし）
    public enum LatentMode {
        case train
        case posterior
        case prior
    }

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
    public func makeBatch(_ samples: [UtteranceSample], mels: [[[Float]]]? = nil, dropLabels: Bool, latent: LatentMode = .posterior, rng: inout SystemRandomNumberGenerator) -> MLXAcousticModel.Batch {
        let cfg = model.config
        let b = samples.count
        let p = samples.map { $0.phoneCount }.max() ?? 1
        let t = samples.map { $0.frameCount }.max() ?? 1
        let f = cfg.featureCount
        let melDim = cfg.melChannels
        let k = cfg.labelKinds.count
        let dropRates = cfg.labelKinds.map { labelDropoutByKind[$0] ?? labelDropout }
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
                if dropLabels && Float.random(in: 0..<1, using: &rng) < dropRates[kk] {
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
            styleMask: MLXArray((0..<b).map { _ -> Float in
                switch latent {
                case .train: return Float.random(in: 0..<1, using: &rng) < styleDropout ? 0 : 1
                case .posterior: return 1
                case .prior: return 0
                }
            }, [b, 1, 1]),
            sampleLatent: latent == .train,
            batchSize: b,
            phoneCount: p,
            frameCount: t
        )
    }

    /// マスク付き損失 [total, mel, dur, pitch, energy, kl, kl の次元ごと]
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
        var total = melLoss * melWeight + durLoss * durationWeight + pitchLoss * pitchWeight + energyLoss * energyWeight
        var kl = MLXArray(Float(0))
        var klPerDim = MLXArray.zeros([1])
        if let mu = pred.latentMean, let logVar = pred.latentLogVar {
            // KL(q(z|mel) || N(0, I)) を次元ごとにバッチ平均し、free bits 未満は罰しない
            klPerDim = mean(0.5 * (square(mu) + exp(logVar) - 1.0 - logVar), axis: 0)
            kl = sum(klPerDim)
            total = total + klWeight * sum(maximum(klPerDim, MLXArray(freeBits)))
        }
        return [total, melLoss, durLoss, pitchLoss, energyLoss, kl, klPerDim]
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
        lastKLPerDim = values[6].asArray(Float.self)
        return Self.toLosses(values)
    }

    /// 勾配無しの評価
    public func evaluate(_ batch: MLXAcousticModel.Batch) -> AcousticLosses {
        let values = losses(model, batch)
        eval(values)
        return Self.toLosses(values)
    }

    static func toLosses(_ v: [MLXArray]) -> AcousticLosses {
        return AcousticLosses(total: v[0].item(Float.self), mel: v[1].item(Float.self), duration: v[2].item(Float.self), pitch: v[3].item(Float.self), energy: v[4].item(Float.self), kl: v[5].item(Float.self))
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
