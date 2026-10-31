import Foundation

/// 合成音と、同じ文の実際の録音とを比べる採点器（CPU のみ）
///
/// 録音側は学習サンプルと同じ作り方の値（アライメントの継続時間、音素ごとのピッチ中央値・エネルギー、実測 Mel）を使う。
/// 合成側は同じ文を同じ音素列で合成するので、整列（DTW）なしで音素ごとに比べられる。
/// 継続時間は予測どおりに合成した結果（自由合成）で、Mel の距離は録音の継続時間を与えて合成した結果で測る。
/// 比はすべて 合成 ÷ 録音（1 が理想）。
public struct ProsodyScore: Sendable, Codable {
    public var id: String
    /// 文頭・文末の無音を除いた総フレーム数の比
    public var durationRatio: Float
    /// 音素ごとの |log(合成 ÷ 録音)| の平均（休止を除く）
    public var durationLogError: Float
    /// 休止（文中の無音）の占める割合: 合成と録音
    public var pauseShare: Float
    public var referencePauseShare: Float
    /// 高さの幅の比（有声音素の半音の標準偏差）
    public var pitchRangeRatio: Float
    /// 発話内の平均を除いた高さの誤差（半音、RMSE）と相関
    public var pitchContourError: Float
    public var pitchCorrelation: Float
    /// 平均の高さの差（半音、合成 − 録音）
    public var pitchLevelDifference: Float
    /// 平均を除いた高さの誤差 ÷ 録音の高さの幅。平らな予測で 1、録音どおりで 0。
    /// 幅だけ広げて形が合わない抑揚は幅の比では良く見えるが、こちらでは悪くなる
    public var pitchNormalizedError: Float
    /// 強さの幅の比（音素ごとのエネルギーの標準偏差）と、隣の音素との差の比
    public var energyRangeRatio: Float
    public var energyJumpRatio: Float
    /// 発話内の平均を除いた強さの誤差（RMSE）
    public var energyContourError: Float
    /// 平均を除いた強さの誤差 ÷ 録音の強さの幅
    public var energyNormalizedError: Float
    /// 録音の継続時間で合成した Mel との L1 距離
    public var melDistance: Float
    /// Mel のくっきりさ（有音フレームでの帯域方向の標準偏差）の比。1 より小さいとぼやけている
    public var melContrastRatio: Float
    /// Mel の時間方向の変化（隣のフレームとの差の絶対値の平均、録音で有音のフレーム）の比。
    /// L1 で学んだ Mel は帯域方向の形は保つが時間方向になまるので、ぼやけはこちらに出る
    public var melFluxRatio: Float

    /// ずれをまとめた値（小さいほど録音に近い）。重みの比較・採択用の素朴な合計
    public var composite: Float {
        return abs(log(max(1e-3, durationRatio))) + durationLogError + pitchNormalizedError + energyNormalizedError + abs(log(max(1e-3, melContrastRatio))) + melDistance
    }
}

public enum ProsodyEvaluation {
    /// 1 発話を採点する。音素列が録音側と合わない（語彙の違いなど）ときは nil
    ///
    /// reference: 録音から作った学習サンプル（mel を持つこと）、labelIndices / latent: 合成の条件
    public static func score(model: AcousticModel, reference s: UtteranceSample, sequence: PhonemeSequence, labelIndices: [Int], latent: [Float]? = nil, intonation: Float = 1.0) -> ProsodyScore? {
        let ids = s.phonemeIds.map { Int($0) }
        guard sequence.ids == ids, s.mel.count == s.frameCount, 3 <= ids.count else {
            return nil
        }
        let free = model.synthesize(phonemeIds: ids, features: sequence.features, labelIndices: labelIndices, intonation: intonation, latent: latent)
        let forced = model.synthesize(phonemeIds: ids, features: sequence.features, labelIndices: labelIndices, intonation: intonation, latent: latent, durations: s.durations, maxLeadSilenceFrames: 10_000, maxTrailSilenceFrames: 10_000)
        let vocabulary = PhonemeVocabulary()
        let body = 1..<(ids.count - 1)
        let isPause = ids.map { $0 == PhonemeVocabulary.pauId || $0 == PhonemeVocabulary.silId }

        // 継続時間
        var predTotal: Float = 0
        var realTotal: Float = 0
        var predPause: Float = 0
        var realPause: Float = 0
        var logErrors: [Float] = []
        for p in body {
            let pred = Float(free.durations[p])
            let real = Float(s.durations[p])
            predTotal += pred
            realTotal += real
            if isPause[p] {
                predPause += pred
                realPause += real
            } else {
                logErrors.append(abs(log(max(1, pred) / max(1, real))))
            }
        }

        // 高さ（録音で有声、かつ合成で高さを持つ音素。半音）
        let voiced = ids.indices.filter { s.voiced[$0] && isPause[$0] != true && vocabulary.isUnvoicedConsonant(id: ids[$0]) != true }
        let predPitch = voiced.map { 12 * forced.pitch[$0] }
        let realPitch = voiced.map { 12 * s.pitch[$0] }

        // 強さ（文頭・文末の無音を除く）
        let predEnergy = body.map { forced.energy[$0] }
        let realEnergy = body.map { s.energy[$0] }

        var melSum: Float = 0
        var melCount = 0
        for t in 0..<min(forced.mel.count, s.mel.count) {
            for c in 0..<min(forced.mel[t].count, s.mel[t].count) {
                melSum += abs(forced.mel[t][c] - s.mel[t][c])
                melCount += 1
            }
        }

        return ProsodyScore(
            id: s.id,
            durationRatio: predTotal / max(1, realTotal),
            durationLogError: mean(logErrors),
            pauseShare: predPause / max(1, predTotal),
            referencePauseShare: realPause / max(1, realTotal),
            pitchRangeRatio: standardDeviation(predPitch) / max(1e-3, standardDeviation(realPitch)),
            pitchContourError: centeredRMSE(predPitch, realPitch),
            pitchCorrelation: correlation(predPitch, realPitch),
            pitchLevelDifference: mean(predPitch) - mean(realPitch),
            pitchNormalizedError: centeredRMSE(predPitch, realPitch) / max(1e-3, standardDeviation(realPitch)),
            energyRangeRatio: standardDeviation(predEnergy) / max(1e-3, standardDeviation(realEnergy)),
            energyJumpRatio: meanJump(predEnergy) / max(1e-3, meanJump(realEnergy)),
            energyContourError: centeredRMSE(predEnergy, realEnergy),
            energyNormalizedError: centeredRMSE(predEnergy, realEnergy) / max(1e-3, standardDeviation(realEnergy)),
            melDistance: melSum / Float(max(1, melCount)),
            melContrastRatio: melContrast(free.mel) / max(1e-3, melContrast(s.mel)),
            melFluxRatio: melFlux(forced.mel, speech: s.mel) / max(1e-3, melFlux(s.mel, speech: s.mel))
        )
    }

    /// 発話ごとの採点の平均（比は対数で平均してから戻す）
    public static func average(_ scores: [ProsodyScore]) -> ProsodyScore? {
        guard scores.isEmpty != true else {
            return nil
        }
        func arithmetic(_ key: KeyPath<ProsodyScore, Float>) -> Float {
            return mean(scores.map { $0[keyPath: key] })
        }
        func geometric(_ key: KeyPath<ProsodyScore, Float>) -> Float {
            return exp(mean(scores.map { log(max(1e-3, $0[keyPath: key])) }))
        }
        return ProsodyScore(
            id: "",
            durationRatio: geometric(\.durationRatio),
            durationLogError: arithmetic(\.durationLogError),
            pauseShare: arithmetic(\.pauseShare),
            referencePauseShare: arithmetic(\.referencePauseShare),
            pitchRangeRatio: geometric(\.pitchRangeRatio),
            pitchContourError: arithmetic(\.pitchContourError),
            pitchCorrelation: arithmetic(\.pitchCorrelation),
            pitchLevelDifference: arithmetic(\.pitchLevelDifference),
            pitchNormalizedError: arithmetic(\.pitchNormalizedError),
            energyRangeRatio: geometric(\.energyRangeRatio),
            energyJumpRatio: geometric(\.energyJumpRatio),
            energyContourError: arithmetic(\.energyContourError),
            energyNormalizedError: arithmetic(\.energyNormalizedError),
            melDistance: arithmetic(\.melDistance),
            melContrastRatio: geometric(\.melContrastRatio),
            melFluxRatio: geometric(\.melFluxRatio)
        )
    }

    /// 有音フレーム（最大値が -6 を超える）での、帯域方向の標準偏差の平均
    public static func melContrast(_ mel: [[Float]]) -> Float {
        let speech = mel.filter { ($0.max() ?? -10) > -6 }
        return mean(speech.map { standardDeviation($0) })
    }

    /// 隣のフレームとの差の絶対値の平均（speech で有音のフレームだけ）
    public static func melFlux(_ mel: [[Float]], speech: [[Float]]) -> Float {
        var sum: Float = 0
        var count = 0
        var t = 1
        while t < min(mel.count, speech.count) {
            if (speech[t].max() ?? -10) > -6 {
                for c in 0..<min(mel[t].count, mel[t - 1].count) {
                    sum += abs(mel[t][c] - mel[t - 1][c])
                }
                count += mel[t].count
            }
            t += 1
        }
        return sum / Float(max(1, count))
    }

    public static func mean(_ v: [Float]) -> Float {
        return v.isEmpty ? 0 : v.reduce(0, +) / Float(v.count)
    }

    static func standardDeviation(_ v: [Float]) -> Float {
        let m = mean(v)
        return mean(v.map { ($0 - m) * ($0 - m) }).squareRoot()
    }

    static func centeredRMSE(_ a: [Float], _ b: [Float]) -> Float {
        let ma = mean(a)
        let mb = mean(b)
        return mean(zip(a, b).map { x, y in (x - ma - (y - mb)) * (x - ma - (y - mb)) }).squareRoot()
    }

    static func correlation(_ a: [Float], _ b: [Float]) -> Float {
        let sa = standardDeviation(a)
        let sb = standardDeviation(b)
        if a.count < 2 || sa < 1e-6 || sb < 1e-6 {
            return 0
        }
        let ma = mean(a)
        let mb = mean(b)
        return mean(zip(a, b).map { ($0 - ma) * ($1 - mb) }) / (sa * sb)
    }

    static func meanJump(_ v: [Float]) -> Float {
        if v.count < 2 {
            return 0
        }
        var sum: Float = 0
        for k in 1..<v.count {
            sum += abs(v[k] - v[k - 1])
        }
        return sum / Float(v.count - 1)
    }
}

/// 採点に使う固定の評価セット（学習の検証側に回した発話から選ぶ）
public struct EvaluationSet: Sendable, Codable {
    public struct Entry: Sendable, Codable, Equatable {
        public var corpus: String
        public var id: String

        public init(corpus: String, id: String) {
            self.corpus = corpus
            self.id = id
        }
    }

    public var entries: [Entry]

    public init(entries: [Entry]) {
        self.entries = entries
    }

    public var ids: Set<String> {
        return Set(entries.map { $0.id })
    }

    /// 候補（コーパスごとの発話 ID と話者）から count 件を選ぶ
    ///
    /// コーパスごとに件数の比で枠を分け（各コーパス最低 minPerCorpus 件）、コーパス内では話者を順に回して 1 件ずつ取る。
    /// 同じ候補からは常に同じ集合を選ぶ。
    public static func select(candidates: [(corpus: String, id: String, speaker: String)], count: Int, minPerCorpus: Int = 20) -> EvaluationSet {
        var corpusOrder: [String] = []
        var byCorpus: [String: [(id: String, speaker: String)]] = [:]
        for c in candidates {
            if byCorpus[c.corpus] == nil {
                corpusOrder.append(c.corpus)
            }
            byCorpus[c.corpus, default: []].append((c.id, c.speaker))
        }
        let total = max(1, candidates.count)
        var quota: [String: Int] = [:]
        for corpus in corpusOrder {
            let size = byCorpus[corpus]!.count
            quota[corpus] = min(size, max(minPerCorpus, count * size / total))
        }
        // 最低枠で増えた分は、いちばん大きいコーパスから減らす
        var excess = quota.values.reduce(0, +) - count
        for corpus in corpusOrder.sorted(by: { byCorpus[$0]!.count > byCorpus[$1]!.count }) where 0 < excess {
            let cut = min(excess, quota[corpus]! - min(quota[corpus]!, minPerCorpus))
            quota[corpus]! -= cut
            excess -= cut
        }
        var entries: [Entry] = []
        for corpus in corpusOrder {
            var speakerOrder: [String] = []
            var bySpeaker: [String: [String]] = [:]
            for item in byCorpus[corpus]! {
                if bySpeaker[item.speaker] == nil {
                    speakerOrder.append(item.speaker)
                }
                bySpeaker[item.speaker, default: []].append(item.id)
            }
            var taken = 0
            var round = 0
            while taken < quota[corpus]! {
                var progressed = false
                for speaker in speakerOrder where taken < quota[corpus]! {
                    let list = bySpeaker[speaker]!
                    if round < list.count {
                        entries.append(Entry(corpus: corpus, id: list[round]))
                        taken += 1
                        progressed = true
                    }
                }
                if progressed != true {
                    break
                }
                round += 1
            }
        }
        return EvaluationSet(entries: entries)
    }
}
