import Foundation
import Mozu

/// 評価セットの作成と、録音との比較による採点
extension TrainCommands {
    /// 学習と同じ手順で学習サンプルを並べ、検証側に回る発話（validationEvery 件に 1 件）から評価セットを選ぶ
    static func runEvalSet(_ args: [String]) -> Int32 {
        guard let a = Args.parse(args), let corpora = loadCorpora(a), a.input != nil, let out = a.output else {
            print("evalset: -d <corpus>、-w <in.json>（学習に使う重み）、-o <set.json> が必要です。")
            return 1
        }
        do {
            let weights = try prepareWeights(a, corpora: corpora)
            var lazy = a
            lazy.lazyMel = true
            let samples = try buildSamples(lazy, corpora: corpora, weights: weights, keepPCM: false)
            var owner: [String: CorpusUtterance] = [:]
            var corpusOf: [String: String] = [:]
            for corpus in corpora {
                for u in corpus.utterances {
                    owner[u.id] = u
                    corpusOf[u.id] = corpus.directory
                }
            }
            var candidates: [(corpus: String, id: String, speaker: String)] = []
            for (i, s) in samples.enumerated() where a.validationEvery > 0 && i % a.validationEvery == 0 {
                let corpus = corpusOf[s.id] ?? ""
                candidates.append((corpus, s.id, owner[s.id]?.labels["speaker"] ?? corpus))
            }
            let set = EvaluationSet.select(candidates: candidates, count: a.count)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(set).write(to: URL(fileURLWithPath: out))
            let perCorpus = Dictionary(grouping: set.entries, by: { $0.corpus }).map { "\($0.key) \($0.value.count) 件" }.sorted()
            let speakers = Set(set.entries.map { owner[$0.id]?.labels["speaker"] ?? $0.corpus }).count
            print("検証側 \(candidates.count) 発話から評価セット \(set.entries.count) 発話（話者 \(speakers) 人）を選び、\(out) に保存しました: " + perCorpus.joined(separator: "、"))
            return 0
        } catch {
            print("evalset: 失敗しました: \(error)")
            return 1
        }
    }

    /// 評価セットの各発話を、録音と同じ条件（ラベル、z = 0）で合成して採点する
    static func runEvaluate(_ args: [String]) -> Int32 {
        guard let a = Args.parse(args), let setPath = a.evalSet, a.models.isEmpty != true else {
            print("evaluate: --set <set.json> と -m <weights.json>（1 つ以上）が必要です。")
            return 1
        }
        do {
            let set = try JSONDecoder().decode(EvaluationSet.self, from: Data(contentsOf: URL(fileURLWithPath: setPath)))
            let models = try a.models.map { try MozuWeights.load(from: URL(fileURLWithPath: $0)) }
            for (path, w) in zip(a.models, models) where w.acoustic == nil {
                print("evaluate: \(path) に音響モデルがありません。")
                return 1
            }
            // 録音側の値は 1 つ目の重みの音素列で作る（他の重みは音素列が同じ発話だけを採点する）
            let base = MozuEngine(weights: models[0])
            var references: [UtteranceSample] = []
            var utterances: [String: CorpusUtterance] = [:]
            var corpusOf: [String: String] = [:]
            for directory in uniqueOrdered(set.entries.map { $0.corpus }) {
                let wanted = Set(set.entries.filter { $0.corpus == directory }.map { $0.id })
                let full = try Corpus.load(directory: directory)
                let corpus = Corpus(directory: full.directory, utterances: full.utterances.filter { wanted.contains($0.id) }, readings: full.readings)
                let alignments = try AlignmentBuilder.loadCached(corpus: corpus, sequenceBuilder: base.sequenceBuilder)
                let built = SampleBuilder.build(corpus: corpus, alignments: alignments, labels: models[0].labels, sequenceBuilder: base.sequenceBuilder, options: SampleBuilder.Options(maxFrames: a.maxFrames, keepMel: true))
                if built.count < wanted.count {
                    print("  \(directory): \(wanted.count - built.count) 発話は音素列がアライメントと合わないため除きました")
                }
                references.append(contentsOf: built)
                for u in corpus.utterances {
                    utterances[u.id] = u
                    corpusOf[u.id] = directory
                }
            }
            print("評価セット \(set.entries.count) 発話のうち \(references.count) 発話を採点します。")

            var results: [[ProsodyScore?]] = []
            for (path, weights) in zip(a.models, models) {
                let engine = MozuEngine(weights: weights)
                let acoustic = engine.acoustic!
                var scores = [ProsodyScore?](repeating: nil, count: references.count)
                let lock = NSLock()
                DispatchQueue.concurrentPerform(iterations: references.count) { i in
                    let s = references[i]
                    // 録音側の音素列には、アライメントで見つかった句読点の無い休止が入っていることがある
                    guard let sequence = AlignmentBuilder.merge(engine.sequenceBuilder.build(text: s.text), into: s.phonemeIds.map { Int($0) }) else {
                        return
                    }
                    let labels = weights.labels.indices(for: utterances[s.id]?.labels ?? [:])
                    // --posterior: 録音から推定した話し方 z を与える（足りないのが話し方の情報かを切り分ける）
                    let latent = a.posterior ? weights.acoustic?.style.map { StyleEncoder(weights: $0).encode(mel: s.mel).mean } : nil
                    let score = ProsodyEvaluation.score(model: acoustic, reference: s, sequence: sequence, labelIndices: labels, latent: latent, intonation: a.intonation)
                    lock.lock()
                    scores[i] = score
                    lock.unlock()
                }
                let missing = scores.filter { $0 == nil }.count
                if 0 < missing {
                    print("  \(path): \(missing) 発話は音素列が録音側と異なるため採点できませんでした")
                }
                results.append(scores)
            }
            // すべての重みで採点できた発話だけで比べる
            let common = references.indices.filter { i in results.allSatisfy { $0[i] != nil } }
            print("共通に採点できた発話: \(common.count) 件\n")

            func groupKey(_ kind: String, _ id: String) -> String {
                if kind == "corpus" {
                    return URL(fileURLWithPath: corpusOf[id] ?? "").lastPathComponent
                }
                return utterances[id]?.labels[kind] ?? "-"
            }
            var groups: [(title: String, members: [Int])] = [("全体", common)]
            for kind in a.groupBy {
                let keys = uniqueOrdered(common.map { groupKey(kind, references[$0].id) }).sorted()
                for key in keys {
                    groups.append(("\(kind)=\(key)", common.filter { groupKey(kind, references[$0].id) == key }))
                }
            }
            let names = a.models.map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent }
            let columns = ["件数", "長さ比", "長さ誤", "休止", "高幅比", "高誤差", "高相関", "高差", "高正誤", "強幅比", "強跳比", "強正誤", "Mel距", "鮮明比", "変化比", "合計"]
            print("グループ / 重み\t" + columns.joined(separator: "\t"))
            for group in groups where group.members.isEmpty != true {
                for (m, name) in names.enumerated() {
                    guard let avg = ProsodyEvaluation.average(group.members.map { results[m][$0]! }) else {
                        continue
                    }
                    let composite = ProsodyEvaluation.mean(group.members.map { results[m][$0]!.composite })
                    let values = [String(group.members.count), String(format: "%.3f", avg.durationRatio), String(format: "%.3f", avg.durationLogError), String(format: "%.2f/%.2f", avg.pauseShare, avg.referencePauseShare), String(format: "%.3f", avg.pitchRangeRatio), String(format: "%.2f", avg.pitchContourError), String(format: "%.3f", avg.pitchCorrelation), String(format: "%+.2f", avg.pitchLevelDifference), String(format: "%.3f", avg.pitchNormalizedError), String(format: "%.3f", avg.energyRangeRatio), String(format: "%.3f", avg.energyJumpRatio), String(format: "%.3f", avg.energyNormalizedError), String(format: "%.3f", avg.melDistance), String(format: "%.3f", avg.melContrastRatio), String(format: "%.3f", avg.melFluxRatio), String(format: "%.3f", composite)]
                    print("\(group.title) / \(name)\t" + values.joined(separator: "\t"))
                }
            }
            print("""

            長さ比: 自由合成の総フレーム数 ÷ 録音（文頭・文末の無音を除く）／長さ誤: 音素ごとの |log 比| の平均／休止: 文中の無音の割合（合成/録音）
            高幅比: 抑揚の幅の比（半音の標準偏差）／高誤差: 平均を除いた高さの誤差（半音）／高相関: 高さの相関／高差: 平均の高さの差（半音）／高正誤: 高誤差 ÷ 録音の抑揚の幅（平らで 1、一致で 0）
            強幅比・強跳比・強正誤: 強さの幅の比、隣の音素との差の比、正規化誤差／Mel距: 録音の継続時間で合成した Mel の L1／鮮明比: Mel のくっきりさ（帯域方向）の比／変化比: Mel の時間方向の変化の比（なまりで 1 より小さくなる）
            合計: |log 長さ比| + 長さ誤 + 高正誤 + 強正誤 + |log 鮮明比| + Mel距（小さいほど録音に近い）
            """)
            if let out = a.output {
                var report: [String: [ProsodyScore]] = [:]
                for (m, name) in names.enumerated() {
                    report[name] = common.map { results[m][$0]! }
                }
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(to: URL(fileURLWithPath: out))
                print("発話ごとの採点を \(out) に保存しました。")
            }
            return 0
        } catch {
            print("evaluate: 失敗しました: \(error)")
            return 1
        }
    }

    static func uniqueOrdered(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { seen.insert($0).inserted }
    }
}
