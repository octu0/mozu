import Foundation

/// 音響モデルへ渡す音素列
public struct PhonemeSequence: Sendable, Equatable {
    public let tokens: [PhonemeToken]
    /// [phoneCount][PhonemeSequenceBuilder.featureCount]
    public let features: [[Float]]

    public var ids: [Int] {
        return tokens.map { $0.id }
    }

    public var count: Int {
        return tokens.count
    }
}

/// テキスト → 音素列（文頭・文末の無音、句間のポーズを含む）とアクセント・位置特徴
///
/// 特徴は [モーラ高低, 句内位置 (0..1), アクセント核, 句頭モーラ, 句末モーラ, 疑問文] の 6 次元。
/// 無音・ポーズの音素はすべて 0。
public struct PhonemeSequenceBuilder: Sendable {
    public static let featureCount: Int = 6

    public let normalizer: TextNormalizer
    public let prosodyModel: ProsodyModel
    public let vocabulary: PhonemeVocabulary

    public init(normalizer: TextNormalizer, prosodyModel: ProsodyModel, vocabulary: PhonemeVocabulary) {
        self.normalizer = normalizer
        self.prosodyModel = prosodyModel
        self.vocabulary = vocabulary
    }

    public func phrases(for text: String) -> [AccentPhrase] {
        let morphemes = normalizer.normalize(text: text)
        return prosodyModel.buildAccentPhrases(morphemes: morphemes, vocabulary: vocabulary)
    }

    public func build(text: String) -> PhonemeSequence {
        return build(phrases: phrases(for: text))
    }

    public func build(phrases: [AccentPhrase]) -> PhonemeSequence {
        var tokens: [PhonemeToken] = []
        var features: [[Float]] = []
        let zero = [Float](repeating: 0.0, count: Self.featureCount)
        let silence = PhonemeToken(id: PhonemeVocabulary.silId, symbol: vocabulary.token(for: PhonemeVocabulary.silId), category: .pause)
        let pause = PhonemeToken(id: PhonemeVocabulary.pauId, symbol: vocabulary.token(for: PhonemeVocabulary.pauId), category: .pause)
        tokens.append(silence)
        features.append(zero)
        var pIdx = 0
        while pIdx < phrases.count {
            let phrase = phrases[pIdx]
            let moraCount = max(1, phrase.moras.count)
            var mIdx = 0
            for mora in phrase.moras {
                let feat: [Float] = [
                    mora.tone == .high ? 1.0 : 0.0,
                    Float(mIdx) / Float(moraCount),
                    mora.isAccentKernel ? 1.0 : 0.0,
                    mIdx == 0 ? 1.0 : 0.0,
                    mIdx == phrase.moras.count - 1 ? 1.0 : 0.0,
                    phrase.isQuestion ? 1.0 : 0.0,
                ]
                for phoneme in mora.phonemes {
                    if phoneme.category == .pause {
                        // 読み由来のポーズは句のポーズとして扱う
                        continue
                    }
                    tokens.append(PhonemeToken(id: phoneme.id, symbol: phoneme.symbol, category: phoneme.category))
                    features.append(feat)
                }
                mIdx += 1
            }
            if phrase.pauseAfter && pIdx < phrases.count - 1 {
                tokens.append(pause)
                features.append(zero)
            }
            pIdx += 1
        }
        tokens.append(silence)
        features.append(zero)
        return PhonemeSequence(tokens: tokens, features: features)
    }
}
