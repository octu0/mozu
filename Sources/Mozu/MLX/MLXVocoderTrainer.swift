#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

public struct VocoderLosses: Sendable {
    public var total: Float = 0
    public var mel: Float = 0
    public var convergence: Float = 0
    public var logMagnitude: Float = 0
    public var adversarial: Float = 0
    public var featureMatching: Float = 0
    public var discriminator: Float = 0

    public static func + (a: VocoderLosses, b: VocoderLosses) -> VocoderLosses {
        return VocoderLosses(total: a.total + b.total, mel: a.mel + b.mel, convergence: a.convergence + b.convergence, logMagnitude: a.logMagnitude + b.logMagnitude, adversarial: a.adversarial + b.adversarial, featureMatching: a.featureMatching + b.featureMatching, discriminator: a.discriminator + b.discriminator)
    }

    public func scaled(_ s: Float) -> VocoderLosses {
        return VocoderLosses(total: total * s, mel: mel * s, convergence: convergence * s, logMagnitude: logMagnitude * s, adversarial: adversarial * s, featureMatching: featureMatching * s, discriminator: discriminator * s)
    }

    public var description: String {
        if adversarial != 0 || discriminator != 0 {
            return String(format: "total %.4f (mel %.4f, sc %.4f, logmag %.4f, adv %.4f, fm %.4f, D %.4f)", total, mel, convergence, logMagnitude, adversarial, featureMatching, discriminator)
        }
        return String(format: "total %.4f (mel %.4f, sc %.4f, logmag %.4f)", total, mel, convergence, logMagnitude)
    }

    public var selection: Float {
        return mel
    }
}

/// ボコーダの学習
///
/// 段階 1: Mel 再構成 + 複数解像度 STFT 損失。段階 2（`discriminator` を与えたとき）: 敵対的損失と特徴一致を加える。
/// 位相は振幅だけの損失では決まらず、GAN が無いと「ぼやけた・こもった」音になるため段階 2 が本番の構成。
public final class MLXVocoderTrainer: @unchecked Sendable {
    public let model: MLXVocoder
    public let optimizer: Adam
    public let melAnalyzer: MLXMelAnalyzer
    public let stftLoss: MLXMultiResolutionSTFTLoss
    public var discriminator: MLXVocoderDiscriminator? = nil
    public var discriminatorOptimizer: Adam? = nil
    public var melWeight: Float = 1.0
    public var convergenceWeight: Float = 1.0
    public var logMagnitudeWeight: Float = 1.0
    /// Vocos の比（Mel 45 : 敵対 1 : 特徴一致 1）に合わせた重み
    public var adversarialWeight: Float = 0.022
    public var featureMatchingWeight: Float = 0.022
    public var gradientClip: Float = 5.0
    public private(set) var lastGradientNorm: Float = 0

    public init(model: MLXVocoder, learningRate: Float) {
        self.model = model
        self.optimizer = Adam(learningRate: learningRate)
        self.melAnalyzer = MLXMelAnalyzer()
        self.stftLoss = MLXMultiResolutionSTFTLoss()
    }

    public var learningRate: Float {
        get { optimizer.learningRate }
        set {
            optimizer.learningRate = newValue
            discriminatorOptimizer?.learningRate = newValue
        }
    }

    public func enableDiscriminator(_ d: MLXVocoderDiscriminator, learningRate: Float) {
        discriminator = d
        discriminatorOptimizer = Adam(learningRate: learningRate, betas: (0.8, 0.99))
    }

    /// 固定長の切り出し（Mel フレーム数 `frames`、PCM frames·hop サンプル）
    public struct Batch {
        public let mel: MLXArray   // [B, L, melChannels]
        public let pcm: MLXArray   // [B, L·hop]
    }

    public func makeBatch(mels: [[[Float]]], pcms: [[Float]], frames: Int, rng: inout SystemRandomNumberGenerator) -> Batch {
        let b = mels.count
        let hop = model.config.hopSize
        let melDim = model.config.melChannels
        var melBuf = [Float](repeating: 0, count: b * frames * melDim)
        var pcmBuf = [Float](repeating: 0, count: b * frames * hop)
        var i = 0
        while i < b {
            let mel = mels[i]
            let pcm = pcms[i]
            let available = min(mel.count, pcm.count / hop)
            let start = available > frames ? Int.random(in: 0...(available - frames), using: &rng) : 0
            var t = 0
            while t < frames {
                let src = start + t
                if src < available {
                    var c = 0
                    while c < melDim {
                        melBuf[(i * frames + t) * melDim + c] = mel[src][c]
                        c += 1
                    }
                    var s = 0
                    while s < hop {
                        pcmBuf[(i * frames + t) * hop + s] = pcm[src * hop + s]
                        s += 1
                    }
                } else {
                    var c = 0
                    while c < melDim {
                        melBuf[(i * frames + t) * melDim + c] = log(MelSpectrogramExtractor.logMelFloor)
                        c += 1
                    }
                }
                t += 1
            }
            i += 1
        }
        return Batch(mel: MLXArray(melBuf, [b, frames, melDim]), pcm: MLXArray(pcmBuf, [b, frames * hop]))
    }

    /// 生成器の損失 [total, mel, sc, logmag, adv, fm]
    func losses(_ m: MLXVocoder, _ batch: Batch, useDiscriminator: Bool) -> [MLXArray] {
        let predicted = m.synthesize(batch.mel)
        let melLoss = mean(abs(melAnalyzer.logMel(predicted) - melAnalyzer.logMel(batch.pcm)))
        let stft = stftLoss.losses(predicted: predicted, target: batch.pcm)
        var total = melLoss * melWeight + stft.convergence * convergenceWeight + stft.logMagnitude * logMagnitudeWeight
        var adv = MLXArray(Float(0))
        var fm = MLXArray(Float(0))
        if useDiscriminator, let d = discriminator {
            let g = d.generatorLosses(real: batch.pcm, fake: predicted)
            adv = g.adversarial
            fm = g.featureMatching
            total = total + adv * adversarialWeight + fm * featureMatchingWeight
        }
        return [total, melLoss, stft.convergence, stft.logMagnitude, adv, fm]
    }

    public func trainStep(_ batch: Batch) -> VocoderLosses {
        var dLoss: Float = 0
        if let d = discriminator, let dOpt = discriminatorOptimizer {
            // 識別器の更新（生成波形は定数として扱う）
            let fake = stopGradient(model.synthesize(batch.mel))
            let dLossFn = { (disc: MLXVocoderDiscriminator, _: [MLXArray]) -> [MLXArray] in
                return [disc.discriminatorLoss(real: batch.pcm, fake: fake)]
            }
            let dValueAndGrad = valueAndGrad(model: d, dLossFn)
            let (dValues, dGrads) = dValueAndGrad(d, [])
            let (dClipped, _) = clipGradNorm(gradients: dGrads, maxNorm: gradientClip)
            dOpt.update(model: d, gradients: dClipped)
            eval(d, dOpt, dValues)
            dLoss = dValues[0].item(Float.self)
        }
        let useD = discriminator != nil
        let lossFn = { (m: MLXVocoder, _: [MLXArray]) -> [MLXArray] in
            return self.losses(m, batch, useDiscriminator: useD)
        }
        let valueAndGradient = valueAndGrad(model: model, lossFn)
        let (values, grads) = valueAndGradient(model, [])
        let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: gradientClip)
        optimizer.update(model: model, gradients: clipped)
        eval(model, optimizer, values, norm)
        lastGradientNorm = norm.item(Float.self)
        var losses = Self.toLosses(values)
        losses.discriminator = dLoss
        return losses
    }

    public func evaluate(_ batch: Batch) -> VocoderLosses {
        let values = losses(model, batch, useDiscriminator: false)
        eval(values)
        return Self.toLosses(values)
    }

    static func toLosses(_ v: [MLXArray]) -> VocoderLosses {
        return VocoderLosses(total: v[0].item(Float.self), mel: v[1].item(Float.self), convergence: v[2].item(Float.self), logMagnitude: v[3].item(Float.self), adversarial: v[4].item(Float.self), featureMatching: v[5].item(Float.self))
    }
}

/// ボコーダ学習の設定
public struct VocoderTrainingOptions: Sendable {
    public var epochs: Int = 10
    public var batchSize: Int = 16
    /// 切り出し長（Mel フレーム数）
    public var segmentFrames: Int = 96
    public var learningRateMax: Float = 3e-4
    public var learningRateMin: Float = 3e-5
    public var warmupSteps: Int = 200
    public var validationEvery: Int = 20
    public var logEvery: Int = 50
    public var seed: UInt64 = 2026
    /// 敵対的学習を使う
    public var useGAN: Bool = false
    /// 敵対的学習を始めるエポック（それまでは再構成損失だけで生成器を温める）
    public var ganStartEpoch: Int = 1

    public init() {}
}

/// ボコーダ学習用のデータ（音声は使うときに読む。大規模コーパスを全部メモリに置かないため）
public struct VocoderDataset: Sendable {
    public let utterances: [CorpusUtterance]
    public init(utterances: [CorpusUtterance]) {
        self.utterances = utterances
    }

    /// 指定した発話の PCM と Mel を並列に読む
    public func load(indices: [Int]) -> (mels: [[[Float]]], pcms: [[Float]]) {
        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor()
        var mels = [[[Float]]?](repeating: nil, count: indices.count)
        var pcms = [[Float]?](repeating: nil, count: indices.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: indices.count) { i in
            guard let pcm = Corpus.loadPCM(path: utterances[indices[i]].wavPath, reader: reader) else {
                return
            }
            let mel = extractor.extractLogMel(pcm: pcm)
            lock.lock()
            mels[i] = mel
            pcms[i] = pcm
            lock.unlock()
        }
        var outMels: [[[Float]]] = []
        var outPcms: [[Float]] = []
        var i = 0
        while i < indices.count {
            if let m = mels[i], let p = pcms[i], m.count >= 8 {
                outMels.append(m)
                outPcms.append(p)
            }
            i += 1
        }
        return (outMels, outPcms)
    }
}

public enum VocoderTraining {
    public static func run(
        dataset: VocoderDataset,
        initial: VocoderWeights?,
        config: VocoderConfig,
        options: VocoderTrainingOptions,
        log: @escaping (String) -> Void,
        onBest: (VocoderWeights, Int, VocoderLosses) -> Void
    ) -> VocoderWeights? {
        if dataset.utterances.isEmpty {
            log("学習データがありません。")
            return nil
        }
        GPU.set(cacheLimit: 4 << 30)
        var rng = SystemRandomNumberGenerator()
        let model: MLXVocoder
        switch initial {
        case .some(let w):
            model = MLXVocoder(weights: w)
            log("既存のボコーダからウォームスタートします（パラメータ \(w.parameterCount)）。")
        case .none:
            model = MLXVocoder(config: config)
            log("ボコーダを新規に初期化しました。")
        }
        let trainer = MLXVocoderTrainer(model: model, learningRate: options.learningRateMin)
        var trainIdx: [Int] = []
        var validIdx: [Int] = []
        var i = 0
        while i < dataset.utterances.count {
            if options.validationEvery > 0 && i % options.validationEvery == 0 {
                validIdx.append(i)
            } else {
                trainIdx.append(i)
            }
            i += 1
        }
        if validIdx.isEmpty {
            validIdx = Array(trainIdx.prefix(min(8, trainIdx.count)))
        }
        let stepsPerEpoch = (trainIdx.count + options.batchSize - 1) / options.batchSize
        let totalSteps = stepsPerEpoch * options.epochs
        let scheduler = CosineLRScheduler(lrMax: options.learningRateMax, lrMin: options.learningRateMin)
        log("学習 \(trainIdx.count) 発話 / 検証 \(validIdx.count) 発話、\(stepsPerEpoch) ステップ/エポック、\(options.epochs) エポック、切り出し \(options.segmentFrames) フレーム")

        // 検証は固定の切り出し位置で比較できるよう、先頭から切り出す
        let validData = dataset.load(indices: validIdx)
        var best: VocoderWeights? = nil
        var bestSelection = Float.greatestFiniteMagnitude
        var step = 0
        var epoch = 1
        while epoch <= options.epochs {
            if options.useGAN && trainer.discriminator == nil && options.ganStartEpoch <= epoch {
                trainer.enableDiscriminator(MLXVocoderDiscriminator(), learningRate: trainer.learningRate)
                log("敵対的学習（MPD + MRD）を開始します。")
            }
            var order = trainIdx
            order.shuffle(using: &rng)
            var epochSum = VocoderLosses()
            var recent = VocoderLosses()
            var recentCount = 0
            var batches = 0
            let started = Date()
            var start = 0
            while start < order.count {
                let end = min(order.count, start + options.batchSize)
                let data = dataset.load(indices: Array(order[start..<end]))
                start = end
                if data.mels.isEmpty {
                    continue
                }
                step += 1
                trainer.learningRate = scheduler.learningRate(step: step, totalSteps: totalSteps, warmupSteps: options.warmupSteps)
                let batch = trainer.makeBatch(mels: data.mels, pcms: data.pcms, frames: options.segmentFrames, rng: &rng)
                let losses = trainer.trainStep(batch)
                if losses.total.isFinite != true {
                    log("警告: 非有限の損失が出ました（ステップ \(step)）。")
                    continue
                }
                batches += 1
                epochSum = epochSum + losses
                recent = recent + losses
                recentCount += 1
                if options.logEvery > 0 && recentCount % options.logEvery == 0 {
                    log(String(format: "  ステップ %d/%d  lr %.2e  勾配ノルム %.2f  直近 %@", batches, stepsPerEpoch, trainer.learningRate, trainer.lastGradientNorm, recent.scaled(1.0 / Float(recentCount)).description))
                    recent = VocoderLosses()
                    recentCount = 0
                }
            }
            let trainAverage = epochSum.scaled(1.0 / Float(max(1, batches)))
            var validSum = VocoderLosses()
            var validBatches = 0
            var vStart = 0
            var fixedRng = SystemRandomNumberGenerator()
            while vStart < validData.mels.count {
                let vEnd = min(validData.mels.count, vStart + options.batchSize)
                let batch = trainer.makeBatch(mels: Array(validData.mels[vStart..<vEnd]), pcms: Array(validData.pcms[vStart..<vEnd]), frames: options.segmentFrames, rng: &fixedRng)
                validSum = validSum + trainer.evaluate(batch)
                validBatches += 1
                vStart = vEnd
            }
            let validAverage = validSum.scaled(1.0 / Float(max(1, validBatches)))
            log(String(format: "[Epoch %d/%d] 学習 %@  (%.0f 秒, GPU ピーク %.1f GB)", epoch, options.epochs, trainAverage.description, Date().timeIntervalSince(started), Double(GPU.peakMemory) / Double(1 << 30)))
            log(String(format: "              検証 %@", validAverage.description))
            if validAverage.selection < bestSelection {
                bestSelection = validAverage.selection
                let weights = model.exportWeights()
                best = weights
                onBest(weights, epoch, validAverage)
            }
            epoch += 1
        }
        return best
    }
}
#endif
