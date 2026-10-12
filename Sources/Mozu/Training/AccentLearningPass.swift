import Foundation

/// 教師 F0 からアクセント核を学習して語彙へ反映する
///
/// 各アクセント句について、モーラの母音部の F0（半音）の並びから「どのモーラの直後で落ちるか」を推定し、
/// 句頭の形態素に投票する。投票は AccentLearning が集計し、語彙の核を更新する。
public enum AccentLearningPass {
    public struct Report: Sendable {
        public var phrases = 0
        public var votes = 0
        public var abstained = 0
        public var mismatched = 0
        public var updated = 0
        public var changed = 0
        public var overridden: [(surface: String, from: Int, to: Int, votes: Int)] = []
        public var consistency: Float = 0
        public var consistentWords = 0
        public var histogram: [Int: Int] = [:]
    }

    /// ProsodyModel.buildAccentPhrases と同じ句切りで、各句の句頭形態素（表記とモーラ数）を求める
    static func phraseInitialMorphemes(_ morphemes: [Morpheme], vocabulary: PhonemeVocabulary) -> [(surface: String, moraCount: Int)] {
        var result: [(surface: String, moraCount: Int)] = []
        var currentEmpty = true
        var current: (surface: String, moraCount: Int)? = nil
        for m in morphemes {
            if m.pos == .symbol {
                if currentEmpty != true, let c = current {
                    result.append(c)
                }
                currentEmpty = true
                current = nil
                continue
            }
            var startsNew = false
            switch m.pos {
            case .noun, .verb, .adjective, .conjunction, .interjection:
                if currentEmpty != true {
                    startsNew = true
                }
            default:
                break
            }
            if startsNew, let c = current {
                result.append(c)
                currentEmpty = true
                current = nil
            }
            let moraCount = vocabulary.kanaToMoras(m.reading).filter { $0.phonemes.contains { $0.category != .pause } }.count
            if currentEmpty {
                current = (surface: m.surface, moraCount: moraCount)
            }
            if 0 < moraCount {
                currentEmpty = false
            }
        }
        if currentEmpty != true, let c = current {
            result.append(c)
        }
        return result
    }

    /// サンプル列から核を学習し、更新した語彙と報告を返す
    public static func run(samples: [UtteranceSample], lexicon: [LexiconEntry], builder: PhonemeSequenceBuilder) -> (lexicon: [LexiconEntry], report: Report) {
        var learning = AccentLearning()
        var report = Report()
        var votesByWord: [String: [Int: Int]] = [:]
        for sample in samples {
            let morphemes = builder.normalizer.normalize(text: sample.text)
            let phrases = builder.prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: builder.vocabulary)
            let initials = phraseInitialMorphemes(morphemes, vocabulary: builder.vocabulary)
            let sequence = builder.build(phrases: phrases)
            if initials.count != phrases.count || sequence.ids.map({ Int32($0) }) != sample.phonemeIds {
                report.mismatched += 1
                continue
            }
            // 音素列の走査: 先頭の無音の次から
            var phone = 1
            var pIdx = 0
            while pIdx < phrases.count {
                let phrase = phrases[pIdx]
                var moraSamples: [AccentEstimator.MoraSample] = []
                for mora in phrase.moras {
                    let phonemes = mora.phonemes.filter { $0.category != .pause }
                    var vowelIndex = -1
                    for _ in phonemes {
                        vowelIndex = phone
                        phone += 1
                    }
                    if 0 <= vowelIndex && vowelIndex < sample.phoneCount && sample.voiced[vowelIndex] {
                        moraSamples.append(AccentEstimator.MoraSample(semitone: 12.0 * sample.pitch[vowelIndex], voicedFrames: sample.voicedFrames[vowelIndex]))
                    } else {
                        moraSamples.append(AccentEstimator.MoraSample(semitone: nil, voicedFrames: 0))
                    }
                }
                if phrase.pauseAfter && pIdx < phrases.count - 1 {
                    phone += 1
                }
                report.phrases += 1
                let initial = initials[pIdx]
                let isFinal = phrase.pauseAfter || pIdx == phrases.count - 1
                switch AccentEstimator.estimateKernel(moras: moraSamples, excludeFinalMora: isFinal) {
                case .kernel(let kernel, let confidence):
                    if kernel <= initial.moraCount {
                        learning.record(surface: initial.surface, kernel: kernel, confidence: confidence)
                        votesByWord[initial.surface, default: [:]][kernel, default: 0] += 1
                        report.votes += 1
                    } else {
                        report.abstained += 1
                    }
                case .abstain:
                    report.abstained += 1
                }
                pIdx += 1
            }
        }
        let applied = learning.apply(to: lexicon)
        report.updated = applied.updated
        report.changed = applied.changed
        report.overridden = applied.overridden
        var shareSum: Float = 0
        var shareCount = 0
        for (_, votes) in votesByWord {
            let total = votes.values.reduce(0, +)
            if 3 <= total, let best = votes.values.max() {
                shareSum += Float(best) / Float(total)
                shareCount += 1
            }
        }
        report.consistency = shareCount > 0 ? shareSum / Float(shareCount) : 0
        report.consistentWords = shareCount
        report.histogram = AccentLearning.kernelHistogram(applied.lexicon)
        return (applied.lexicon, report)
    }
}
