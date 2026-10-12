#if canImport(MLX)
import Foundation
import MLX

/// 音響モデル学習の設定
public struct AcousticTrainingOptions: Sendable {
    public var epochs: Int = 10
    public var batchSize: Int = 16
    public var learningRateMax: Float = 5e-4
    public var learningRateMin: Float = 5e-5
    public var warmupSteps: Int = 200
    /// 検証に回す発話の間隔（20 なら 20 件に 1 件）
    public var validationEvery: Int = 20
    public var seed: UInt64 = 2026
    public var labelDropout: Float = 0.1
    public var logEvery: Int = 50
    /// 1 バッチの詰め物込み総フレーム数の上限（メモリの山を抑える）
    public var maxBatchFrames: Int = 12000
    /// MLX の Metal バッファキャッシュの上限（バイト）
    public var gpuCacheLimit: Int = 4 << 30

    public init() {}
}

/// 音響モデルの既定構成
public enum AcousticDefaults {
    public static func config(labelKinds: [String], modelDim: Int = 256, encoderLayers: Int = 4, decoderLayers: Int = 6) -> AcousticConfig {
        let encoder = HawkStackConfig(modelDim: modelDim, rnnDim: modelDim, mlpDim: modelDim * 3, heads: 4, convLeft: 2, convRight: 2, numLayers: encoderLayers, bidirectional: true)
        let decoder = HawkStackConfig(modelDim: modelDim, rnnDim: modelDim, mlpDim: modelDim * 3, heads: 4, convLeft: 2, convRight: 2, numLayers: decoderLayers, bidirectional: true)
        return AcousticConfig(
            phonemeVocabSize: PhonemeVocabulary().size,
            featureCount: PhonemeSequenceBuilder.featureCount,
            modelDim: modelDim,
            headHiddenDim: 128,
            melChannels: AudioConfig.melChannels,
            labelKinds: labelKinds,
            encoder: encoder,
            decoder: decoder,
            pitchMean: 0, pitchStd: 1, energyMean: 0, energyStd: 1
        )
    }
}

/// 音響モデルの学習ループ
public enum AcousticTraining {
    public static func run(
        samples: [UtteranceSample],
        labels: LabelVocabulary,
        initial: AcousticWeights?,
        config: AcousticConfig,
        options: AcousticTrainingOptions,
        log: @escaping (String) -> Void,
        onBest: (AcousticWeights, Int, AcousticLosses) -> Void
    ) -> AcousticWeights? {
        if samples.isEmpty {
            log("学習サンプルがありません。")
            return nil
        }
        GPU.set(cacheLimit: options.gpuCacheLimit)
        var rng = SystemRandomNumberGenerator()
        // 正規化定数は毎回データから取り直す（ウォームスタートでもコーパスが変われば変わる）
        let norm = SampleBuilder.normalization(samples: samples)
        let effectiveConfig = (initial?.config ?? config).withNormalization(pitchMean: norm.pitchMean, pitchStd: norm.pitchStd, energyMean: norm.energyMean, energyStd: norm.energyStd)
        let labelCounts = effectiveConfig.labelKinds.map { labels.count(kind: $0) }
        let model: MLXAcousticModel
        switch initial {
        case .some(let w):
            model = MLXAcousticModel(config: effectiveConfig, labelCounts: labelCounts)
            model.load(w)
            // ラベル語彙が増えていたら埋め込み表を広げる
            var k = 0
            while k < labelCounts.count {
                let old = model.labelEmbeddings[k]
                if old.shape[0] < labelCounts[k] {
                    model.labelEmbeddings[k] = concatenated([old, MLXArray.zeros([labelCounts[k] - old.shape[0], effectiveConfig.modelDim])], axis: 0)
                }
                k += 1
            }
            log("既存の音響モデルからウォームスタートします（パラメータ \(w.parameterCount)）。")
        case .none:
            model = MLXAcousticModel(config: effectiveConfig, labelCounts: labelCounts)
            log("音響モデルを新規に初期化しました。")
        }
        let trainer = MLXAcousticTrainer(model: model, learningRate: options.learningRateMin)
        trainer.labelDropout = options.labelDropout

        var train: [UtteranceSample] = []
        var valid: [UtteranceSample] = []
        var i = 0
        while i < samples.count {
            if options.validationEvery > 0 && i % options.validationEvery == 0 {
                valid.append(samples[i])
            } else {
                train.append(samples[i])
            }
            i += 1
        }
        if valid.isEmpty {
            valid = Array(train.prefix(min(8, train.count)))
        }
        let stepsPerEpoch = BatchPlanner.plan(sampleFrames: train.map { $0.frameCount }, batchSize: options.batchSize, maxFrames: options.maxBatchFrames, rng: &rng).count
        let totalSteps = stepsPerEpoch * options.epochs
        let scheduler = CosineLRScheduler(lrMax: options.learningRateMax, lrMin: options.learningRateMin)
        log(String(format: "学習 %d 件 / 検証 %d 件、%d ステップ/エポック、%d エポック、ピッチ正規化 (%.3f, %.3f)、エネルギー正規化 (%.3f, %.3f)", train.count, valid.count, stepsPerEpoch, options.epochs, norm.pitchMean, norm.pitchStd, norm.energyMean, norm.energyStd))

        var best: AcousticWeights? = nil
        var bestSelection = Float.greatestFiniteMagnitude
        var step = 0
        var epoch = 1
        while epoch <= options.epochs {
            let batches = BatchPlanner.plan(sampleFrames: train.map { $0.frameCount }, batchSize: options.batchSize, maxFrames: options.maxBatchFrames, rng: &rng)
            var epochSum = AcousticLosses()
            var recent = AcousticLosses()
            var recentCount = 0
            let started = Date()
            for indices in batches {
                step += 1
                trainer.learningRate = scheduler.learningRate(step: step, totalSteps: totalSteps, warmupSteps: options.warmupSteps)
                let chosen = indices.map { train[$0] }
                let mels = chosen.contains { $0.mel.isEmpty } ? SampleBuilder.loadMels(samples: chosen) : nil
                let batch = trainer.makeBatch(chosen, mels: mels, dropLabels: true, rng: &rng)
                let losses = trainer.trainStep(batch)
                if losses.total.isFinite != true {
                    log("警告: 非有限の損失が出ました（ステップ \(step)）。")
                    continue
                }
                epochSum = epochSum + losses
                recent = recent + losses
                recentCount += 1
                if options.logEvery > 0 && recentCount % options.logEvery == 0 {
                    log(String(format: "  ステップ %d/%d  lr %.2e  勾配ノルム %.2f  直近 %@", step - (epoch - 1) * stepsPerEpoch, stepsPerEpoch, trainer.learningRate, trainer.lastGradientNorm, recent.scaled(1.0 / Float(recentCount)).description))
                    recent = AcousticLosses()
                    recentCount = 0
                }
            }
            let trainAverage = epochSum.scaled(1.0 / Float(max(1, batches.count)))
            // 検証（ラベルは落とさない）
            var validSum = AcousticLosses()
            var validBatches = 0
            var vStart = 0
            while vStart < valid.count {
                let vEnd = min(valid.count, vStart + options.batchSize)
                let chosen = Array(valid[vStart..<vEnd])
                let mels = chosen.contains { $0.mel.isEmpty } ? SampleBuilder.loadMels(samples: chosen) : nil
                let batch = trainer.makeBatch(chosen, mels: mels, dropLabels: false, rng: &rng)
                validSum = validSum + trainer.evaluate(batch)
                validBatches += 1
                vStart = vEnd
            }
            let validAverage = validSum.scaled(1.0 / Float(max(1, validBatches)))
            let elapsed = Date().timeIntervalSince(started)
            log(String(format: "[Epoch %d/%d] 学習 %@  (%.0f 秒, GPU ピーク %.1f GB)", epoch, options.epochs, trainAverage.description, elapsed, Double(GPU.peakMemory) / Double(1 << 30)))
            log(String(format: "              検証 %@  採択値 %.4f", validAverage.description, validAverage.selection))
            if validAverage.selection < bestSelection {
                bestSelection = validAverage.selection
                let weights = model.exportWeights(config: effectiveConfig)
                best = weights
                onBest(weights, epoch, validAverage)
            }
            epoch += 1
        }
        return best
    }
}
#endif
