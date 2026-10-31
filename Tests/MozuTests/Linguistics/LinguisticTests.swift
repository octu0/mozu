import XCTest
@testable import Mozu

/// 日本語言語処理フロントエンド（形態素解析, 読み正規化, 語彙, 韻律, ）の網羅的単体テストスイート
final class LinguisticTests: XCTestCase {

    private var normalizer: TextNormalizer!
    private var morphology: ViterbiMorphology!
    private var vocabulary: PhonemeVocabulary!
    private var prosodyModel: ProsodyModel!

    override func setUp() {
        super.setUp()
        self.morphology = ViterbiMorphology()
        self.normalizer = TextNormalizer(morphology: morphology)
        self.vocabulary = PhonemeVocabulary()
        self.prosodyModel = ProsodyModel()
    }

    // MARK: - 1. 形態素解析・品詞特定テスト

    func testViterbiMorphologicalAnalysis() {
        let morphemes = morphology.tokenize("これは水です")
        XCTAssertEqual(morphemes.count, 4)

        XCTAssertEqual(morphemes[0].surface, "これ")
        XCTAssertEqual(morphemes[0].pos, .noun)

        XCTAssertEqual(morphemes[1].surface, "は")
        XCTAssertEqual(morphemes[1].pos, .particle)

        XCTAssertEqual(morphemes[2].surface, "水")
        XCTAssertEqual(morphemes[2].pos, .noun)

        XCTAssertEqual(morphemes[3].surface, "です")
        XCTAssertEqual(morphemes[3].pos, .auxiliaryVerb)
    }

    // MARK: - 2. 助詞読み替えテスト (は/へ/を)

    func testParticleReadingReplacement() {
        let text1 = "これは水です"
        let norm1 = normalizer.normalize(text: text1)
        let reading1 = norm1.map { $0.reading }.joined()
        XCTAssertEqual(reading1, "これわみずです")

        let text2 = "東京へ行く"
        let norm2 = normalizer.normalize(text: text2)
        let reading2 = norm2.map { $0.reading }.joined()
        XCTAssertEqual(reading2, "とーきょーえいく")

        let text3 = "本を読む"
        let norm3 = normalizer.normalize(text: text3)
        let reading3 = norm3.map { $0.reading }.joined()
        XCTAssertEqual(reading3, "ほんおよむ")

        let text4 = "母は花を愛す"
        let norm4 = normalizer.normalize(text: text4)
        let reading4 = norm4.map { $0.reading }.joined()
        XCTAssertEqual(reading4, "ははわはなおあいす")
    }

    // MARK: - 3. 動詞終止形「う」保護および母音連続長音化テスト

    func testVerbEndingUProtectionAndProlongation() {
        let textVerb1 = "彼を思う"
        let normVerb1 = normalizer.normalize(text: textVerb1)
        let readingVerb1 = normVerb1.map { $0.reading }.joined()
        XCTAssertEqual(readingVerb1, "かれおおもう")

        let textVerb2 = "靴を買う"
        let normVerb2 = normalizer.normalize(text: textVerb2)
        let readingVerb2 = normVerb2.map { $0.reading }.joined()
        XCTAssertEqual(readingVerb2, "くつおかう")

        let textNoun1 = "東京の先生"
        let normNoun1 = normalizer.normalize(text: textNoun1)
        let readingNoun1 = normNoun1.map { $0.reading }.joined()
        XCTAssertEqual(readingNoun1, "とーきょーのせんせー")

        let textNoun2 = "映画を見る"
        let normNoun2 = normalizer.normalize(text: textNoun2)
        let readingNoun2 = normNoun2.map { $0.reading }.joined()
        XCTAssertEqual(readingNoun2, "えーがおみる")
    }

    // MARK: - 4. 万進法数詞展開および助数詞連声テスト

    func testNumeralExpansionAndCounterSandhi() {
        let resYear = normalizer.expandNumbersAndCounters("1473年")
        XCTAssertEqual(resYear, "せんよんひゃくななじゅうさんねん")

        let resHon1 = normalizer.expandNumbersAndCounters("1本")
        XCTAssertEqual(resHon1, "いっぽん")

        let resHon3 = normalizer.expandNumbersAndCounters("3本")
        XCTAssertEqual(resHon3, "さんぼん")

        let resHon10 = normalizer.expandNumbersAndCounters("10本")
        XCTAssertEqual(resHon10, "じゅっぽん")

        let resMonth = normalizer.expandNumbersAndCounters("4月")
        XCTAssertEqual(resMonth, "しがつ")

        let resDay = normalizer.expandNumbersAndCounters("4日")
        XCTAssertEqual(resDay, "よっか")

        let resHour = normalizer.expandNumbersAndCounters("4時")
        XCTAssertEqual(resHour, "よじ")

        let resPerson1 = normalizer.expandNumbersAndCounters("1人")
        XCTAssertEqual(resPerson1, "ひとり")

        let resPerson2 = normalizer.expandNumbersAndCounters("2人")
        XCTAssertEqual(resPerson2, "ふたり")
    }

    // MARK: - 5. モーラ・音素階層トークナイズおよび 64 語彙 ID テスト

    func testMoraAndPhonemeTokenization() {
        let morasKyo = vocabulary.kanaToMoras("きょう")
        XCTAssertEqual(morasKyo.count, 2)
        XCTAssertEqual(morasKyo[0].text, "きょ")
        XCTAssertEqual(morasKyo[0].phonemes.count, 2)
        XCTAssertEqual(morasKyo[0].phonemes[0].symbol, "ky")
        XCTAssertEqual(morasKyo[0].phonemes[1].symbol, "o")
        XCTAssertEqual(morasKyo[1].text, "う")

        let morasGakko = vocabulary.kanaToMoras("がっこう")
        XCTAssertEqual(morasGakko.count, 4)
        XCTAssertEqual(morasGakko[1].phonemes[0].symbol, "Q")
        XCTAssertEqual(morasGakko[1].phonemes[0].category, .geminate)

        for mora in morasGakko {
            for p in mora.phonemes {
                XCTAssertTrue(0 <= p.id)
                XCTAssertTrue(p.id < 64)
            }
        }
    }

    // MARK: - 6. 東京方言ピッチアクセントトーンおよび F0 輪郭生成テスト

    func testPhonemeHelperHy() {
        let hyId = 32

        XCTAssertEqual(vocabulary.token(for: hyId), "hy", "ID 32 は hy であること")
        XCTAssertTrue(vocabulary.isUnvoicedFricative(id: hyId), "hy は無声摩擦音であること")
        XCTAssertTrue(vocabulary.isUnvoicedConsonant(id: hyId), "hy は無声子音であること")
        XCTAssertTrue(vocabulary.isVoiced(symbol: "hy") != true, "hy は無声音であること")
    }

    // MARK: - 7. 転写からの語彙獲得

    func testLexiconLearningAcquiresKatakanaWords() {
        let result = ViterbiMorphology.learnLexiconEntries(sentences: ["コップに水を入れる。", "ア、そう。"], existing: [])
        let surfaces = Set(result.lexicon.filter { $0.cost == ViterbiMorphology.learnedLexiconCost }.map { $0.surface })
        XCTAssertTrue(surfaces.contains("コップ"), "2 文字以上のカタカナ語を獲得すること")
        XCTAssertEqual(result.lexicon.first { $0.surface == "コップ" }?.reading, "こっぷ")
        XCTAssertTrue(surfaces.contains("水"), "漢字語も従来どおり獲得すること")
        XCTAssertTrue(surfaces.contains("ア") != true, "1 文字のカタカナは獲得しないこと")
    }

    // MARK: - 8. 語彙に無いひらがな列

    /// 語彙に無いひらがなの拗音は前の文字と合わせて 1 拍で読み、1 文字ずつ別の語に分けない
    func testUnknownHiraganaKeepsContractedSounds() {
        let builder = PhonemeSequenceBuilder(normalizer: normalizer, prosodyModel: prosodyModel, vocabulary: vocabulary)
        func symbols(_ text: String) -> [String] {
            return builder.build(text: text).tokens.map { $0.symbol }.filter { $0 != "<sil>" }
        }
        XCTAssertEqual(symbols("おっしゃー"), ["o", "Q", "sh", "a", "_"])
        XCTAssertEqual(symbols("しゃ"), ["sh", "a"])
        // 1 語として 1 つのアクセント句になる（拍ごとに句頭にならない）
        let phraseStarts = builder.build(text: "おっしゃー").features.filter { $0[3] == 1 }.count
        XCTAssertEqual(phraseStarts, 1)
        // 小書き文字から始まる形態素を作らない
        for text in ["ちょっとまって", "いらっしゃい", "しゃべりすぎ"] {
            let small: Set<Character> = ["ゃ", "ゅ", "ょ"]
            XCTAssertFalse(normalizer.normalize(text: text).contains { $0.surface.first.map { small.contains($0) } ?? false }, text)
        }
    }

    /// 語の境界をまたいだ連結語は取り除き、複合語は残す。かなを含む語どうしは連結して獲得しない
    func testLexiconLearningDropsWordsJoinedAcrossBoundaries() {
        let cost = ViterbiMorphology.learnedLexiconCost
        let existing = [
            LexiconEntry(surface: "遠足", reading: "えんそく", pos: .noun, accentKernel: 0, cost: cost),
            LexiconEntry(surface: "行け", reading: "いけ", pos: .noun, accentKernel: 0, cost: cost),
            LexiconEntry(surface: "遠足行け", reading: "えんそくいけ", pos: .noun, accentKernel: 1, cost: cost),
            LexiconEntry(surface: "夏", reading: "なつ", pos: .noun, accentKernel: 2, cost: cost),
            LexiconEntry(surface: "休み", reading: "やすみ", pos: .noun, accentKernel: 3, cost: cost),
            LexiconEntry(surface: "夏休み", reading: "なつやすみ", pos: .noun, accentKernel: 3, cost: cost),
        ]
        let result = ViterbiMorphology.learnLexiconEntries(sentences: [], existing: existing)
        let surfaces = Set(result.lexicon.map { $0.surface })
        XCTAssertFalse(surfaces.contains("遠足行け"))
        XCTAssertTrue(surfaces.contains("夏休み"))
        XCTAssertEqual(result.removed, 1)

        let learned = ViterbiMorphology.learnLexiconEntries(sentences: ["明日は遠足行けないから寂しい。"], existing: [])
        XCTAssertFalse(learned.lexicon.contains { $0.surface.hasPrefix("遠足") && $0.surface.count > 2 }, "かなを含む語どうしを連結しない")
    }
}
