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
}
