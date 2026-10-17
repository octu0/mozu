import Foundation
import Mozu

/// サブコマンドの実装（学習本体は Mozu ライブラリにあり、ここは引数の解釈と進捗表示だけを持つ）
enum TrainCommands {
    struct Args {
        var corpora: [String] = []
        var input: String? = nil
        var output: String? = nil
        var from: String? = nil
        var epochs: Int = 10
        var batchSize: Int = 16
        var lrMax: Float = 5e-4
        var lrMin: Float = 5e-5
        var warmup: Int = 200
        var limit: Int? = nil
        var force: Bool = false
        var modelDim: Int = 256
        var encoderLayers: Int = 4
        var decoderLayers: Int = 6
        var vocoderLayers: Int = 6
        var validationEvery: Int = 20
        var labelDropout: Float = 0.1
        var seed: UInt64 = 2026
        var maxFrames: Int = 2000
        var noAccentLearning: Bool = false
        var gan: Bool = false
        var ganStartEpoch: Int = 1
        var segmentFrames: Int = 96
        var lazyMel: Bool = false
        var maxBatchFrames: Int = 12000

        static func parse(_ args: [String]) -> Args? {
            var a = Args()
            var i = 0
            func value() -> String? {
                i += 1
                return i < args.count ? args[i] : nil
            }
            while i < args.count {
                switch args[i] {
                case "-d", "--corpus": if let v = value() { a.corpora.append(v) }
                case "-w", "--weights": a.input = value()
                case "-o", "--output": a.output = value()
                case "--from": a.from = value()
                case "-e", "--epochs": a.epochs = Int(value() ?? "") ?? a.epochs
                case "--batch": a.batchSize = Int(value() ?? "") ?? a.batchSize
                case "--lr": a.lrMax = Float(value() ?? "") ?? a.lrMax; if a.lrMin > a.lrMax { a.lrMin = a.lrMax / 10 }
                case "--lr-min": a.lrMin = Float(value() ?? "") ?? a.lrMin
                case "--warmup": a.warmup = Int(value() ?? "") ?? a.warmup
                case "--limit": a.limit = Int(value() ?? "")
                case "--force": a.force = true
                case "--model-dim": a.modelDim = Int(value() ?? "") ?? a.modelDim
                case "--encoder-layers": a.encoderLayers = Int(value() ?? "") ?? a.encoderLayers
                case "--decoder-layers": a.decoderLayers = Int(value() ?? "") ?? a.decoderLayers
                case "--vocoder-layers": a.vocoderLayers = Int(value() ?? "") ?? a.vocoderLayers
                case "--valid-every": a.validationEvery = Int(value() ?? "") ?? a.validationEvery
                case "--label-dropout": a.labelDropout = Float(value() ?? "") ?? a.labelDropout
                case "--seed": a.seed = UInt64(value() ?? "") ?? a.seed
                case "--max-frames": a.maxFrames = Int(value() ?? "") ?? a.maxFrames
                case "--no-accent-learning": a.noAccentLearning = true
                case "--gan": a.gan = true
                case "--lazy-mel": a.lazyMel = true
                case "--max-batch-frames": a.maxBatchFrames = Int(value() ?? "") ?? a.maxBatchFrames
                case "--gan-start": a.ganStartEpoch = Int(value() ?? "") ?? a.ganStartEpoch
                case "--segment-frames": a.segmentFrames = Int(value() ?? "") ?? a.segmentFrames
                default:
                    print("不明な引数: \(args[i])")
                    return nil
                }
                i += 1
            }
            return a
        }
    }

    /// 重みの読み込み（無ければ空）と、コーパスの転写からの語彙獲得
    static func prepareWeights(_ a: Args, corpora: [Corpus]) throws -> MozuWeights {
        var weights = MozuWeights()
        if let path = a.input {
            weights = try MozuWeights.load(from: URL(fileURLWithPath: path))
            print("重みを読み込みました: \(path)（語彙 \(weights.lexicon.count) 件、パラメータ \(weights.parameterCount)）")
        }
        let sentences = corpora.flatMap { $0.utterances.map { $0.text } }
        var readings: [String: String] = [:]
        for corpus in corpora {
            readings.merge(corpus.readings) { first, _ in first }
        }
        let learned = ViterbiMorphology.learnLexiconEntries(sentences: sentences, existing: weights.lexicon, readings: readings)
        if 0 < learned.added {
            weights.lexicon = learned.lexicon
            print("転写 \(learned.sentenceCount) 文から語彙を \(learned.added) 件獲得しました（語彙合計 \(weights.lexicon.count) 件）。")
        } else {
            print("転写 \(learned.sentenceCount) 文に新規語彙はありませんでした（語彙合計 \(weights.lexicon.count) 件）。")
        }
        weights.labels.register(corpora: corpora)
        if weights.labels.kinds.isEmpty != true {
            let summary = weights.labels.kinds.map { "\($0) \(weights.labels.count(kind: $0) - 1) 種" }.joined(separator: ", ")
            print("条件付けラベル: " + summary)
        }
        return weights
    }

    static func loadCorpora(_ a: Args) -> [Corpus]? {
        if a.corpora.isEmpty {
            print("-d <corpus> を 1 つ以上指定してください。")
            return nil
        }
        var corpora: [Corpus] = []
        for dir in a.corpora {
            do {
                let corpus = try Corpus.load(directory: dir, limit: a.limit)
                print("コーパス \(dir): \(corpus.utterances.count) 発話")
                corpora.append(corpus)
            } catch {
                print("コーパス \(dir) を読めません: \(error)")
                return nil
            }
        }
        return corpora
    }

    static func runAlign(_ args: [String]) -> Int32 {
        guard let a = Args.parse(args), let corpora = loadCorpora(a) else {
            return 1
        }
        do {
            let weights = try prepareWeights(a, corpora: corpora)
            let engine = MozuEngine(weights: weights)
            for corpus in corpora {
                let result = try AlignmentBuilder.loadOrBuild(corpus: corpus, sequenceBuilder: engine.sequenceBuilder, force: a.force)
                print("アライメント: キャッシュ \(result.statistics.loaded) 件、生成 \(result.statistics.built) 件、除外 \(result.statistics.skipped) 件")
            }
            if let out = a.output {
                try weights.save(to: URL(fileURLWithPath: out))
                print("語彙を含む重みを保存しました: \(out)")
            }
            return 0
        } catch {
            print("align: 失敗しました: \(error)")
            return 1
        }
    }

    static func buildSamples(_ a: Args, corpora: [Corpus], weights: MozuWeights, keepPCM: Bool) throws -> [UtteranceSample] {
        let engine = MozuEngine(weights: weights)
        let total = corpora.reduce(0) { $0 + $1.utterances.count }
        let keepMel = a.lazyMel != true && total <= 8000
        if keepMel != true {
            print("Mel はメモリに置かず、バッチごとに読み直します（\(total) 発話）。")
        }
        var samples: [UtteranceSample] = []
        for corpus in corpora {
            let result = try AlignmentBuilder.loadOrBuild(corpus: corpus, sequenceBuilder: engine.sequenceBuilder, force: a.force)
            print("アライメント \(corpus.directory): キャッシュ \(result.statistics.loaded) 件、生成 \(result.statistics.built) 件、除外 \(result.statistics.skipped) 件")
            let built = SampleBuilder.build(corpus: corpus, alignments: result.alignments, labels: weights.labels, sequenceBuilder: engine.sequenceBuilder, options: SampleBuilder.Options(keepPCM: keepPCM, maxFrames: a.maxFrames, keepMel: keepMel))
            print("学習サンプル \(corpus.directory): \(built.count) 件（\(corpus.utterances.count - built.count) 件除外）")
            samples.append(contentsOf: built)
        }
        return samples
    }

    static func runAcoustic(_ args: [String]) -> Int32 {
        guard let a = Args.parse(args), let corpora = loadCorpora(a), let out = a.output else {
            print("acoustic: -d <corpus> と -o <out.json> が必要です。")
            return 1
        }
        do {
            var weights = try prepareWeights(a, corpora: corpora)
            var samples = try buildSamples(a, corpora: corpora, weights: weights, keepPCM: false)
            if a.noAccentLearning != true {
                samples = learnAccents(samples: samples, weights: &weights)
            }
            var options = AcousticTrainingOptions()
            options.epochs = a.epochs
            options.batchSize = a.batchSize
            options.learningRateMax = a.lrMax
            options.learningRateMin = a.lrMin
            options.warmupSteps = a.warmup
            options.validationEvery = a.validationEvery
            options.labelDropout = a.labelDropout
            options.seed = a.seed
            options.maxBatchFrames = a.maxBatchFrames
            let config = AcousticDefaults.config(labelKinds: weights.labels.kinds, modelDim: a.modelDim, encoderLayers: a.encoderLayers, decoderLayers: a.decoderLayers)
            let outURL = URL(fileURLWithPath: out)
            let result = AcousticTraining.run(samples: samples, labels: weights.labels, initial: weights.acoustic, config: config, options: options, log: { print($0) }) { best, epoch, losses in
                weights.acoustic = best
                do {
                    try weights.save(to: outURL)
                    print("  ==> 最良更新: Epoch \(epoch) 採択値 \(String(format: "%.4f", losses.selection))。\(out) に保存しました。")
                } catch {
                    print("  保存に失敗しました: \(error)")
                }
            }
            if result == nil {
                return 1
            }
            print("学習が完了しました。")
            return 0
        } catch {
            print("acoustic: 失敗しました: \(error)")
            return 1
        }
    }

    /// 教師 F0 からアクセント核を学習し、語彙とサンプルのアクセント特徴を更新する
    static func learnAccents(samples: [UtteranceSample], weights: inout MozuWeights) -> [UtteranceSample] {
        let engine = MozuEngine(weights: weights)
        let result = AccentLearningPass.run(samples: samples, lexicon: weights.lexicon, builder: engine.sequenceBuilder)
        let r = result.report
        print("アクセント核の学習: 句 \(r.phrases) 件、投票 \(r.votes) 件、保留 \(r.abstained) 件、音素列不一致で除外 \(r.mismatched) 文。語彙更新 \(r.updated) 語（核が変化 \(r.changed) 語）。")
        for o in r.overridden {
            print("  既存の核を話者の実現形で上書き: \(o.surface) 核\(o.from) → 核\(o.to)（\(o.votes) 票）")
        }
        print(String(format: "  投票の一貫性: 3 票以上の語 %d 語で勝者票率の平均 %.2f", r.consistentWords, r.consistency))
        let histogram = r.histogram.keys.sorted().map { "核\($0)=\(r.histogram[$0] ?? 0)" }.joined(separator: " ")
        print("  語彙の核分布: " + histogram)
        if r.changed == 0 {
            return samples
        }
        weights.lexicon = result.lexicon
        let updated = MozuEngine(weights: weights)
        var replaced = 0
        let rebuilt = samples.map { sample -> UtteranceSample in
            let sequence = updated.sequenceBuilder.build(text: sample.text)
            if sequence.ids.map({ Int32($0) }) == sample.phonemeIds {
                replaced += 1
                return sample.withFeatures(sequence.features)
            }
            return sample
        }
        print("  学習サンプルのアクセント特徴を再計算しました: \(replaced)/\(samples.count) 件")
        return rebuilt
    }

    static func runVocoder(_ args: [String]) -> Int32 {
        guard let a = Args.parse(args), let corpora = loadCorpora(a), let out = a.output else {
            print("vocoder: -d <corpus> と -o <out.json> が必要です。")
            return 1
        }
        do {
            var weights = MozuWeights()
            if let path = a.input {
                weights = try MozuWeights.load(from: URL(fileURLWithPath: path))
                print("重みを読み込みました: \(path)（パラメータ \(weights.parameterCount)）")
            }
            let dataset = VocoderDataset(utterances: corpora.flatMap { $0.utterances })
            var options = VocoderTrainingOptions()
            options.epochs = a.epochs
            options.batchSize = a.batchSize
            options.learningRateMax = a.lrMax
            options.learningRateMin = a.lrMin
            options.warmupSteps = a.warmup
            options.validationEvery = a.validationEvery
            options.seed = a.seed
            options.useGAN = a.gan
            options.ganStartEpoch = a.ganStartEpoch
            options.segmentFrames = a.segmentFrames
            let config = VocoderDefaults.config(modelDim: a.modelDim, layers: a.vocoderLayers)
            let outURL = URL(fileURLWithPath: out)
            let result = VocoderTraining.run(dataset: dataset, initial: weights.vocoder, config: config, options: options, log: { print($0) }) { best, epoch, losses in
                weights.vocoder = best
                do {
                    try weights.save(to: outURL)
                    print("  ==> 最良更新: Epoch \(epoch) 検証 Mel \(String(format: "%.4f", losses.mel))。\(out) に保存しました。")
                } catch {
                    print("  保存に失敗しました: \(error)")
                }
            }
            if result == nil {
                return 1
            }
            print("学習が完了しました。")
            return 0
        } catch {
            print("vocoder: 失敗しました: \(error)")
            return 1
        }
    }

    static func runLexicon(_ args: [String]) -> Int32 {
        guard let a = Args.parse(args), let fromPath = a.from, let outPath = a.output else {
            print("lexicon: --from <weights.json> と -o <out.json> が必要です。")
            return 1
        }
        do {
            let lexicon = try MozuWeights.loadLexicon(from: URL(fileURLWithPath: fromPath))
            var weights = MozuWeights()
            if let inPath = a.input {
                weights = try MozuWeights.load(from: URL(fileURLWithPath: inPath))
            }
            weights.lexicon = lexicon
            try weights.save(to: URL(fileURLWithPath: outPath))
            print("語彙 \(lexicon.count) 件を \(fromPath) から取り込み、\(outPath) に保存しました。")
            return 0
        } catch {
            print("lexicon: 失敗しました: \(error)")
            return 1
        }
    }
}
