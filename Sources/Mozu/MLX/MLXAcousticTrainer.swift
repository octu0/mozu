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
    /// 抑揚の識別器に対する敵対的損失・特徴一致・識別器自身の損失（識別器が無ければ 0）
    public var adversarial: Float = 0
    public var featureMatching: Float = 0
    public var discriminator: Float = 0
    /// 抑揚の幅の比（発話ごとの有声音素のピッチの標準偏差、予測 ÷ 教師の平均）。1 に近いほど実音声どおり
    public var rangeRatio: Float = 0
    /// Mel の識別器に対する敵対的損失・特徴一致・識別器自身の損失（識別器が無ければ 0）
    public var melAdversarial: Float = 0
    public var melFeatureMatching: Float = 0
    public var melDiscriminator: Float = 0
    /// Mel の時間方向の変化の比（隣のフレームとの差の絶対値の合計、予測 ÷ 実測）。1 より小さいと時間方向になまっている
    public var fluxRatio: Float = 0

    public static func + (a: AcousticLosses, b: AcousticLosses) -> AcousticLosses {
        return AcousticLosses(total: a.total + b.total, mel: a.mel + b.mel, duration: a.duration + b.duration, pitch: a.pitch + b.pitch, energy: a.energy + b.energy, kl: a.kl + b.kl, adversarial: a.adversarial + b.adversarial, featureMatching: a.featureMatching + b.featureMatching, discriminator: a.discriminator + b.discriminator, rangeRatio: a.rangeRatio + b.rangeRatio, melAdversarial: a.melAdversarial + b.melAdversarial, melFeatureMatching: a.melFeatureMatching + b.melFeatureMatching, melDiscriminator: a.melDiscriminator + b.melDiscriminator, fluxRatio: a.fluxRatio + b.fluxRatio)
    }

    public func scaled(_ s: Float) -> AcousticLosses {
        return AcousticLosses(total: total * s, mel: mel * s, duration: duration * s, pitch: pitch * s, energy: energy * s, kl: kl * s, adversarial: adversarial * s, featureMatching: featureMatching * s, discriminator: discriminator * s, rangeRatio: rangeRatio * s, melAdversarial: melAdversarial * s, melFeatureMatching: melFeatureMatching * s, melDiscriminator: melDiscriminator * s, fluxRatio: fluxRatio * s)
    }

    public var description: String {
        var text = String(format: "total %.4f (mel %.4f, dur %.4f, pitch %.4f, energy %.4f", total, mel, duration, pitch, energy)
        if kl != 0 {
            text += String(format: ", kl %.3f", kl)
        }
        if adversarial != 0 || discriminator != 0 {
            text += String(format: ", adv %.3f, fm %.3f, D %.3f", adversarial, featureMatching, discriminator)
        }
        if melAdversarial != 0 || melDiscriminator != 0 {
            text += String(format: ", melAdv %.3f, melFm %.3f, melD %.3f", melAdversarial, melFeatureMatching, melDiscriminator)
        }
        return text + String(format: ", 抑揚比 %.2f, 変化比 %.2f)", rangeRatio, fluxRatio)
    }

    /// 採択に使う値（Mel とピッチ。継続時間は過学習しやすく、採択基準に入れると Mel の学習を捨ててしまう）
    public var selection: Float {
        return mel + pitch
    }

    /// 抑揚の識別器を使う学習での採択値（Mel と、抑揚の幅の比が 1 からどれだけ離れているか）
    ///
    /// なぜ selection を使わないか: ピッチの二乗誤差は平均に寄せた平板な抑揚ほど小さくなるので、抑揚を広げた重みが採択されなくなる
    public var rangeAwareSelection: Float {
        return mel + abs(log(max(rangeRatio, 1e-3)))
    }

    /// Mel の識別器を使う学習での採択値（Mel の L1 と、時間方向の変化の比が 1 からどれだけ離れているか）
    ///
    /// なぜ L1 だけで選ばないか: L1 はなまった Mel ほど小さくなるので、識別器が効いてくるほど採択されなくなる
    public var fluxAwareSelection: Float {
        return mel + abs(log(max(fluxRatio, 1e-3)))
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
    /// 抑揚の識別器（任意）と、その最適化器
    public private(set) var prosodyDiscriminator: MLXProsodyDiscriminator? = nil
    var discriminatorOptimizer: Adam? = nil
    /// 敵対的損失と特徴一致の重み（学習ループが徐々に上げる）
    public var adversarialWeight: Float = 0
    public var featureMatchingWeight: Float = 0
    /// Mel の識別器（任意）と、その最適化器・重み
    public private(set) var melDiscriminator: MLXMelDiscriminator? = nil
    var melDiscriminatorOptimizer: Adam? = nil
    public var melAdversarialWeight: Float = 0
    public var melFeatureMatchingWeight: Float = 0
    /// Mel の識別器に見せる切り出しの長さ（フレーム）。発話全体に 2 次元の畳み込みをかけると長い発話で GPU が止まる
    public var melSegmentFrames: Int = 128

    public func enableProsodyDiscriminator(_ d: MLXProsodyDiscriminator, learningRate: Float) {
        prosodyDiscriminator = d
        discriminatorOptimizer = Adam(learningRate: learningRate, betas: (0.8, 0.99))
    }

    public func enableMelDiscriminator(_ d: MLXMelDiscriminator, learningRate: Float) {
        melDiscriminator = d
        melDiscriminatorOptimizer = Adam(learningRate: learningRate, betas: (0.8, 0.99))
    }

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
        set {
            optimizer.learningRate = newValue
            discriminatorOptimizer?.learningRate = newValue
            melDiscriminatorOptimizer?.learningRate = newValue
        }
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
        var segmentIndex = [Int32](repeating: 0, count: b * min(melSegmentFrames, t))
        var segmentMask = [Float](repeating: 0, count: b * min(melSegmentFrames, t))
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
            // Mel の識別器用の切り出し（発話内のランダムな位置）
            let segment = min(melSegmentFrames, t)
            let start = s.frameCount > segment ? Int.random(in: 0...(s.frameCount - segment), using: &rng) : 0
            var q = 0
            while q < segment {
                let frameInUtterance = start + q
                let valid = frameInUtterance < s.frameCount
                segmentIndex[i * segment + q] = Int32(i * t + (valid ? frameInUtterance : 0))
                segmentMask[i * segment + q] = valid ? 1 : 0
                q += 1
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
            segmentIndex: MLXArray(segmentIndex),
            segmentMask: MLXArray(segmentMask, [b, min(melSegmentFrames, t), 1]),
            segmentFrames: min(melSegmentFrames, t),
            batchSize: b,
            phoneCount: p,
            frameCount: t
        )
    }

    /// 抑揚の並び [B, P, 2]（無声音素のピッチは 0 にそろえ、識別器が無声音素の不定な予測値で見分けないようにする）
    ///
    /// 継続時間は入れない: 入れると識別器に合わせて継続時間が音素ごとに崩れた（長さの誤差 0.30 → 0.35）。
    /// 継続時間ヘッドはエンコーダから勾配を止めており、識別器の勾配だけが継続時間を動かすことになるため
    static func prosody(pitch: MLXArray, energy: MLXArray, voicedMask: MLXArray) -> MLXArray {
        return concatenated([pitch * voicedMask, energy], axis: -1)
    }

    /// 発話ごとの抑揚の幅の比（予測 ÷ 教師、有声音素のピッチの標準偏差）のバッチ平均
    static func rangeRatio(predicted: MLXArray, teacher: MLXArray, voicedMask: MLXArray) -> MLXArray {
        let n = maximum(sum(voicedMask, axis: 1), MLXArray(Float(1)))
        func spread(_ x: MLXArray) -> MLXArray {
            let m = sum(x * voicedMask, axis: 1) / n
            return sqrt(sum(square(x - m.expandedDimensions(axis: 1)) * voicedMask, axis: 1) / n + 1e-6)
        }
        return mean(stopGradient(spread(predicted)) / stopGradient(spread(teacher)))
    }

    /// Mel の識別器に見せる切り出し [B, S, melChannels]
    static func segments(_ mel: MLXArray, _ batch: MLXAcousticModel.Batch) -> MLXArray {
        let c = mel.shape[2]
        return take(mel.reshaped([batch.batchSize * batch.frameCount, c]), batch.segmentIndex, axis: 0).reshaped([batch.batchSize, batch.segmentFrames, c])
    }

    /// Mel の時間方向の変化の比（予測 ÷ 実測、隣り合う 2 フレームがともに有効な所だけ）
    static func fluxRatio(predicted: MLXArray, teacher: MLXArray, frameMask: MLXArray) -> MLXArray {
        let t = frameMask.shape[1]
        if t < 2 {
            return MLXArray(Float(1))
        }
        let pairMask = frameMask[0..., 1...] * frameMask[0..., ..<(t - 1)]
        let p = sum(abs(predicted[0..., 1...] - predicted[0..., ..<(t - 1)]) * pairMask)
        let r = sum(abs(teacher[0..., 1...] - teacher[0..., ..<(t - 1)]) * pairMask)
        return stopGradient(p / maximum(r, MLXArray(Float(1e-6))))
    }

    /// マスク付き損失 [total, mel, dur, pitch, energy, kl, kl の次元ごと, adv, fm, 抑揚比]
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
        var adv = MLXArray(Float(0))
        var fm = MLXArray(Float(0))
        if let d = prosodyDiscriminator, 0 < adversarialWeight || 0 < featureMatchingWeight {
            let real = Self.prosody(pitch: batch.pitch, energy: batch.energy, voicedMask: batch.voicedMask)
            let fake = Self.prosody(pitch: pred.pitch, energy: pred.energy, voicedMask: batch.voicedMask)
            (adv, fm) = d.generatorLosses(encoded: stopGradient(pred.encoded), real: real, fake: fake, phoneMask: batch.phoneMask)
            total = total + adversarialWeight * adv + featureMatchingWeight * fm
        }
        var melAdv = MLXArray(Float(0))
        var melFm = MLXArray(Float(0))
        if let d = melDiscriminator, 0 < melAdversarialWeight || 0 < melFeatureMatchingWeight {
            (melAdv, melFm) = d.generatorLosses(real: Self.segments(batch.mel, batch), fake: Self.segments(pred.mel, batch), frameMask: batch.segmentMask)
            total = total + melAdversarialWeight * melAdv + melFeatureMatchingWeight * melFm
        }
        let ratio = Self.rangeRatio(predicted: pred.pitch, teacher: batch.pitch, voicedMask: batch.voicedMask)
        let flux = Self.fluxRatio(predicted: pred.mel, teacher: batch.mel, frameMask: batch.frameMask)
        return [total, melLoss, durLoss, pitchLoss, energyLoss, kl, klPerDim, adv, fm, ratio, melAdv, melFm, flux]
    }

    /// 1 バッチの更新（識別器があれば先に識別器を 1 回更新する）
    public func trainStep(_ batch: MLXAcousticModel.Batch) -> AcousticLosses {
        var dLoss: Float = 0
        var melDLoss: Float = 0
        // 識別器の更新に使う生成側の出力（勾配は流さない）。識別器があるときだけ 1 回作る
        let detached: MLXAcousticModel.Prediction? = (prosodyDiscriminator != nil || melDiscriminator != nil) ? model.forward(batch) : nil
        if let d = melDiscriminator, let dOpt = melDiscriminatorOptimizer, let pred = detached {
            let fake = stopGradient(Self.segments(pred.mel, batch))
            let real = Self.segments(batch.mel, batch)
            let dLossFn = { (disc: MLXMelDiscriminator, _: [MLXArray]) -> [MLXArray] in
                return [disc.discriminatorLoss(real: real, fake: fake, frameMask: batch.segmentMask)]
            }
            let (dValues, dGrads) = valueAndGrad(model: d, dLossFn)(d, [])
            let (dClipped, _) = clipGradNorm(gradients: dGrads, maxNorm: gradientClip)
            dOpt.update(model: d, gradients: dClipped)
            eval(d, dOpt, dValues)
            melDLoss = dValues[0].item(Float.self)
        }
        if let d = prosodyDiscriminator, let dOpt = discriminatorOptimizer, let pred = detached {
            let encoded = stopGradient(pred.encoded)
            let real = Self.prosody(pitch: batch.pitch, energy: batch.energy, voicedMask: batch.voicedMask)
            let fake = stopGradient(Self.prosody(pitch: pred.pitch, energy: pred.energy, voicedMask: batch.voicedMask))
            let dLossFn = { (disc: MLXProsodyDiscriminator, _: [MLXArray]) -> [MLXArray] in
                return [disc.discriminatorLoss(encoded: encoded, real: real, fake: fake, phoneMask: batch.phoneMask)]
            }
            let (dValues, dGrads) = valueAndGrad(model: d, dLossFn)(d, [])
            let (dClipped, _) = clipGradNorm(gradients: dGrads, maxNorm: gradientClip)
            dOpt.update(model: d, gradients: dClipped)
            eval(d, dOpt, dValues)
            dLoss = dValues[0].item(Float.self)
        }
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
        var losses = Self.toLosses(values)
        losses.discriminator = dLoss
        losses.melDiscriminator = melDLoss
        return losses
    }

    /// 勾配無しの評価
    public func evaluate(_ batch: MLXAcousticModel.Batch) -> AcousticLosses {
        let values = losses(model, batch)
        eval(values)
        return Self.toLosses(values)
    }

    static func toLosses(_ v: [MLXArray]) -> AcousticLosses {
        return AcousticLosses(total: v[0].item(Float.self), mel: v[1].item(Float.self), duration: v[2].item(Float.self), pitch: v[3].item(Float.self), energy: v[4].item(Float.self), kl: v[5].item(Float.self), adversarial: v[7].item(Float.self), featureMatching: v[8].item(Float.self), rangeRatio: v[9].item(Float.self), melAdversarial: v[10].item(Float.self), melFeatureMatching: v[11].item(Float.self), fluxRatio: v[12].item(Float.self))
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
