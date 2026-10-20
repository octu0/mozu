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
    /// 種類ごとのラベルの置き換え確率（例: ["speaker": 0.5]）。指定の無い種類は labelDropout
    public var labelDropoutByKind: [String: Float] = [:]
    /// 話し方の潜在変数（参照エンコーダ）を使う。重みに無ければ新しく作る
    public var useStyle: Bool = false
    /// KL の重みの最大値と、そこまで上げるステップ数
    /// 0.01 では z がほとんど使われず（次元ごとの KL が free bits に張り付く）、0 では分布が崩れるので、その間の値
    public var klWeight: Float = 0.001
    public var klWarmupSteps: Int = 5000
    /// 次元ごとの最低情報量（nats）
    public var freeBits: Float = 0.1
    /// 潜在変数を伏せる確率
    public var styleDropout: Float = 0.2
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
        // 重みに無い種類のラベルがコーパスにあれば末尾に足す（既存の種類の順序と埋め込みは変えない）
        let baseConfig = initial?.config ?? config
        let addedKinds = labels.kinds.filter { baseConfig.labelKinds.contains($0) != true }
        let effectiveConfig = baseConfig.withLabelKinds(baseConfig.labelKinds + addedKinds).withNormalization(pitchMean: norm.pitchMean, pitchStd: norm.pitchStd, energyMean: norm.energyMean, energyStd: norm.energyStd)
        let labelCounts = effectiveConfig.labelKinds.map { labels.count(kind: $0) }
        let model: MLXAcousticModel
        switch initial {
        case .some(let w):
            model = MLXAcousticModel(config: effectiveConfig, labelCounts: labelCounts)
            model.load(w)
            // ラベルの種類や値が増えていたら埋め込み表を足す・広げる（新しい分は 0 = 未指定と同じから始める）
            var k = 0
            while k < labelCounts.count {
                if model.labelEmbeddings.count <= k {
                    model.labelEmbeddings.append(MLXArray.zeros([labelCounts[k], effectiveConfig.modelDim]))
                } else if model.labelEmbeddings[k].shape[0] < labelCounts[k] {
                    let old = model.labelEmbeddings[k]
                    model.labelEmbeddings[k] = concatenated([old, MLXArray.zeros([labelCounts[k] - old.shape[0], effectiveConfig.modelDim])], axis: 0)
                }
                k += 1
            }
            log("既存の音響モデルからウォームスタートします（パラメータ \(w.parameterCount)）。")
            if addedKinds.isEmpty != true {
                log("新しいラベルの種類を追加しました: \(addedKinds.joined(separator: ", "))")
            }
        case .none:
            model = MLXAcousticModel(config: effectiveConfig, labelCounts: labelCounts)
            log("音響モデルを新規に初期化しました。")
        }
        if options.useStyle && model.style == nil {
            model.style = MLXStyleEncoder(melChannels: effectiveConfig.melChannels, modelDim: effectiveConfig.modelDim)
            log("話し方の潜在変数（参照エンコーダ、\(StyleDefaults.latentDim) 次元）を新しく追加しました。")
        }
        let trainer = MLXAcousticTrainer(model: model, learningRate: options.learningRateMin)
        trainer.freeBits = options.freeBits
        trainer.styleDropout = options.styleDropout
        trainer.labelDropout = options.labelDropout
        trainer.labelDropoutByKind = options.labelDropoutByKind
        if options.labelDropoutByKind.isEmpty != true {
            log("ラベルの置き換え確率: 既定 \(options.labelDropout)、" + options.labelDropoutByKind.keys.sorted().map { "\($0) \(options.labelDropoutByKind[$0]!)" }.joined(separator: "、"))
        }

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
                trainer.klWeight = options.klWeight * min(1.0, Float(step) / Float(max(1, options.klWarmupSteps)))
                let chosen = indices.map { train[$0] }
                let mels = chosen.contains { $0.mel.isEmpty } ? SampleBuilder.loadMels(samples: chosen) : nil
                let batch = trainer.makeBatch(chosen, mels: mels, dropLabels: true, latent: .train, rng: &rng)
                let losses = trainer.trainStep(batch)
                if losses.total.isFinite != true {
                    log("警告: 非有限の損失が出ました（ステップ \(step)）。")
                    continue
                }
                epochSum = epochSum + losses
                recent = recent + losses
                recentCount += 1
                if options.logEvery > 0 && recentCount % options.logEvery == 0 {
                    var line = String(format: "  ステップ %d/%d  lr %.2e  勾配ノルム %.2f  直近 %@", step - (epoch - 1) * stepsPerEpoch, stepsPerEpoch, trainer.learningRate, trainer.lastGradientNorm, recent.scaled(1.0 / Float(recentCount)).description)
                    if model.style != nil {
                        // 次元が使われなくなっていないかの確認用（free bits を超えた次元数と、次元ごとの KL）
                        let active = trainer.lastKLPerDim.filter { options.freeBits < $0 }.count
                        line += String(format: "  β %.4f  有効次元 %d/%d [", trainer.klWeight, active, trainer.lastKLPerDim.count) + trainer.lastKLPerDim.map { String(format: "%.2f", $0) }.joined(separator: " ") + "]"
                    }
                    log(line)
                    recent = AcousticLosses()
                    recentCount = 0
                }
            }
            let trainAverage = epochSum.scaled(1.0 / Float(max(1, batches.count)))
            // 検証（ラベルは落とさない）。潜在変数がある場合は z = 0（参照なし）で採択し、推定した z での値も併記する
            func validate(_ latent: MLXAcousticTrainer.LatentMode) -> AcousticLosses {
                var validSum = AcousticLosses()
                var validBatches = 0
                var vStart = 0
                while vStart < valid.count {
                    let vEnd = min(valid.count, vStart + options.batchSize)
                    let chosen = Array(valid[vStart..<vEnd])
                    let mels = chosen.contains { $0.mel.isEmpty } ? SampleBuilder.loadMels(samples: chosen) : nil
                    let batch = trainer.makeBatch(chosen, mels: mels, dropLabels: false, latent: latent, rng: &rng)
                    validSum = validSum + trainer.evaluate(batch)
                    validBatches += 1
                    vStart = vEnd
                }
                return validSum.scaled(1.0 / Float(max(1, validBatches)))
            }
            let validAverage = validate(model.style == nil ? .posterior : .prior)
            let validPosterior: AcousticLosses? = model.style == nil ? nil : validate(.posterior)
            let elapsed = Date().timeIntervalSince(started)
            log(String(format: "[Epoch %d/%d] 学習 %@  (%.0f 秒, GPU ピーク %.1f GB)", epoch, options.epochs, trainAverage.description, elapsed, Double(GPU.peakMemory) / Double(1 << 30)))
            log(String(format: "              検証 %@  採択値 %.4f", validAverage.description, validAverage.selection) + (model.style == nil ? "" : "（z = 0）"))
            if let posterior = validPosterior {
                log("              検証（参照から推定した z） " + posterior.description)
            }
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
