import Foundation

/// 教師音声の F0 からアクセント核をデータ駆動で推定し、語彙（重み内 lexicon）へ反映するための汎用アルゴリズム。
///
/// なぜ必要か:
/// 転写から獲得した語彙はアクセント核が未知（0 = 平板）で登録されるため、学習データのアクセント特徴が
/// ほぼ一定になり、音響モデルはアクセント特徴を無視して平均的な F0 形状しか出せなくなる。
/// 辞書をソースコードに持ち込まず、コーパスの音声そのものからアクセント核を推定して重みの語彙へ蓄積する。
public enum AccentEstimator {
    /// 推定結果
    public enum Verdict: Equatable {
        /// 核の位置（0 = 平板、k = k モーラ目の直後で下降）と確信度（半音単位の下降量）
        case kernel(Int, confidence: Float)
        /// 判定を保留（曖昧）
        case abstain
    }

    /// モーラの代表 F0（母音部の中央値、半音）と有声フレーム数
    public struct MoraSample: Sendable, Equatable {
        public var semitone: Float?
        public var voicedFrames: Int
        public init(semitone: Float?, voicedFrames: Int) {
            self.semitone = semitone
            self.voicedFrames = voicedFrames
        }
    }

    /// アクセント句のモーラ別 F0 から核を推定する。
    ///
    /// 核の定義は「音響的にどのモーラの直後で F0 が落ちるか」。辞書的な核位置とはピーク遅延の分だけ
    /// ずれることがあるが、合成側（ProsodyModel のモーラ別 H/L）と同じ定義で一貫していればよい。
    /// - Parameters:
    ///   - moras: モーラごとの代表 F0（12·log2(Hz)）と有声フレーム数
    ///   - excludeFinalMora: 句末がポーズ・文末のときは最終モーラの句末下降を核と誤認しないよう除外する
    ///   - minVoicedFrames: これ未満の有声フレームしか無いモーラは信頼せず無視する
    ///   - outlierSemitones: 両隣からこれ以上離れた値は追跡誤り（オクターブ誤り等）として無視する
    ///   - dropThreshold: 核と見なす隣接モーラ間の最小下降量（半音）
    ///   - heibanCeiling: これ未満の最大下降なら平板と判定する（半音）
    ///   - margin: 最大下降と 2 番目の下降の差がこれ未満なら保留（半音）
    ///   - peakTolerance: 核モーラが句の最高値からこれ以上低ければ保留（下降は山の頂から始まるはず）
    ///   - implausibleDrop: 隣接モーラ間でこれを超える下降は追跡のオクターブ誤りとみなして保留（半音）
    ///   - confidenceCap: 確信度（投票の重み）の上限。大きすぎる下降ほど信用できるわけではないため。
    public static func estimateKernel(
        moras: [MoraSample],
        excludeFinalMora: Bool,
        minVoicedFrames: Int = 2,
        outlierSemitones: Float = 6.0,
        dropThreshold: Float = 3.5,
        heibanCeiling: Float = 2.0,
        margin: Float = 1.0,
        peakTolerance: Float = 1.5,
        implausibleDrop: Float = 11.0,
        confidenceCap: Float = 8.0
    ) -> Verdict {
        var values: [Float?] = moras.map { m in
            if let v = m.semitone, v.isFinite, minVoicedFrames <= m.voicedFrames {
                return v
            }
            return nil
        }
        if excludeFinalMora && 1 < values.count {
            values.removeLast()
        }
        // 外れ値除去: 両隣（定義済み）から大きく離れた単独の値を捨てる
        var defined: [(index: Int, value: Float)] = []
        var i = 0
        while i < values.count {
            if let v = values[i] {
                defined.append((index: i, value: v))
            }
            i += 1
        }
        // 無声化モーラ直後の短い母音（例: ひ(無声)つ）は追跡開始点で F0 が異常に高く出ることがある。
        // 直前が未定義で有声フレームが少なく、次のモーラより大きく高い先頭値は追跡誤りとして捨てる。
        if 2 <= defined.count {
            let firstIdx = defined[0].index
            let precededByNil = 0 < firstIdx && values[firstIdx - 1] == nil
            let frames = firstIdx < moras.count ? moras[firstIdx].voicedFrames : 0
            if precededByNil && frames < 4 && outlierSemitones < (defined[0].value - defined[1].value) {
                defined.removeFirst()
            }
        }
        if 3 <= defined.count {
            var keep = [Bool](repeating: true, count: defined.count)
            var d = 1
            while d + 1 < defined.count {
                let prev = defined[d - 1].value
                let cur = defined[d].value
                let next = defined[d + 1].value
                if outlierSemitones < abs(cur - prev) && outlierSemitones < abs(cur - next) && ((prev < cur) == (next < cur)) {
                    keep[d] = false
                }
                d += 1
            }
            var filtered: [(index: Int, value: Float)] = []
            d = 0
            while d < defined.count {
                if keep[d] {
                    filtered.append(defined[d])
                }
                d += 1
            }
            defined = filtered
        }
        if defined.count < 2 {
            return .abstain
        }
        var peak: Float = -Float.greatestFiniteMagnitude
        for item in defined where peak < item.value {
            peak = item.value
        }
        var drops: [(kernel: Int, drop: Float, from: Float)] = []
        var d = 0
        while d + 1 < defined.count {
            drops.append((kernel: defined[d].index + 1, drop: defined[d].value - defined[d + 1].value, from: defined[d].value))
            d += 1
        }
        let sorted = drops.sorted { $0.drop > $1.drop }
        let best = sorted[0]
        let second: Float = 1 < sorted.count ? sorted[1].drop : -Float.greatestFiniteMagnitude
        if best.drop < heibanCeiling {
            // 句末モーラを除外した句では、語内に下降が無いことは平板の証拠にならない（下降は除外したモーラに乗る）
            if excludeFinalMora {
                return .abstain
            }
            return .kernel(0, confidence: heibanCeiling - max(best.drop, 0.0))
        }
        if best.drop < dropThreshold {
            return .abstain
        }
        if (best.drop - second) < margin {
            return .abstain
        }
        if peakTolerance < (peak - best.from) {
            return .abstain
        }
        if implausibleDrop < best.drop {
            return .abstain
        }
        return .kernel(best.kernel, confidence: min(confidenceCap, best.drop))
    }

}

/// 表層形ごとのアクセント核の投票を集計し、語彙へ反映する。
public struct AccentLearning {
    public struct Tally: Sendable {
        public var weights: [Int: Float] = [:]
        public var votes: Int = 0
        public var maxConfidence: Float = 0.0
        public mutating func add(kernel: Int, weight: Float) {
            weights[kernel] = (weights[kernel] ?? 0.0) + weight
            votes += 1
        }
        /// 勝者とその重み比率
        public func winner() -> (kernel: Int, share: Float)? {
            var total: Float = 0.0
            var bestK = -1
            var bestW: Float = -1.0
            for (k, w) in weights.sorted(by: { $0.key < $1.key }) {
                total += w
                if bestW < w {
                    bestW = w
                    bestK = k
                }
            }
            if bestK < 0 || total <= 0.0 {
                return nil
            }
            return (kernel: bestK, share: bestW / total)
        }
    }

    public private(set) var tallies: [String: Tally] = [:]
    /// 勝者に必要な重み比率
    public var minimumShare: Float = 0.6

    public init() {}

    /// 1 回の出現を記録する。kernel は語（句頭形態素）のモーラ数以内であること。
    /// 1 票しか無い語に核を与えるために必要な確信度（半音）
    public var singleVoteConfidence: Float = 4.5

    public mutating func record(surface: String, kernel: Int, confidence: Float) {
        let key = surface.precomposedStringWithCanonicalMapping
        var t = tallies[key] ?? Tally()
        t.add(kernel: kernel, weight: max(0.1, confidence))
        t.maxConfidence = max(t.maxConfidence, confidence)
        tallies[key] = t
    }

    /// 強い証拠（票数・勝者比率）があれば既存の核に関わらず上書きするための閾値
    public var strongVotes: Int = 5
    public var strongShare: Float = 0.7

    /// 投票結果で語彙の核を更新した語彙を返す。
    /// - 獲得語彙（cost == learnedLexiconCost、核が未知で 0 のまま登録された語）は通常の閾値で更新する。
    /// - それ以外（手作業の基本語彙）は、票数 strongVotes 以上かつ勝者比率 strongShare 以上のときだけ上書きし、
    ///   上書きした語は overridden に列挙する（辞書の核位置とピーク遅延後の実際の下降位置は 1 モーラずれ得るため、
    ///   話者の実現形に合わせる）。
    public func apply(to lexicon: [LexiconEntry]) -> (lexicon: [LexiconEntry], updated: Int, changed: Int, overridden: [(surface: String, from: Int, to: Int, votes: Int)]) {
        var out: [LexiconEntry] = []
        out.reserveCapacity(lexicon.count)
        var updated = 0
        var changed = 0
        var overridden: [(surface: String, from: Int, to: Int, votes: Int)] = []
        for entry in lexicon {
            let key = entry.surface.precomposedStringWithCanonicalMapping
            guard let tally = tallies[key], let w = tally.winner(), minimumShare <= w.share else {
                out.append(entry)
                continue
            }
            let isLearned = entry.cost == ViterbiMorphology.learnedLexiconCost && entry.accentKernel == 0
            if isLearned {
                if tally.votes < 2 && w.kernel != 0 && tally.maxConfidence < singleVoteConfidence {
                    out.append(entry)
                    continue
                }
            } else {
                if tally.votes < strongVotes || w.share < strongShare || Int(entry.accentKernel) == w.kernel {
                    out.append(entry)
                    continue
                }
                overridden.append((surface: entry.surface, from: Int(entry.accentKernel), to: w.kernel, votes: tally.votes))
            }
            out.append(LexiconEntry(
                surface: entry.surface,
                reading: entry.reading,
                pos: entry.pos,
                accentKernel: Int16(w.kernel),
                cost: entry.cost
            ))
            updated += 1
            if Int(entry.accentKernel) != w.kernel {
                changed += 1
            }
        }
        return (lexicon: out, updated: updated, changed: changed, overridden: overridden)
    }

    /// 更新後の語彙に含まれる核の分布
    public static func kernelHistogram(_ lexicon: [LexiconEntry]) -> [Int: Int] {
        var h: [Int: Int] = [:]
        for e in lexicon {
            h[Int(e.accentKernel)] = (h[Int(e.accentKernel)] ?? 0) + 1
        }
        return h
    }
}
