import XCTest
@testable import Mozu

/// 英字の語の日本語英語読み（綴りの規則）と、形態素解析での扱いを検証する
final class EnglishReadingTests: XCTestCase {
    /// 一般的なカタカナ英語の読みが候補に含まれる（曖昧な語の第 1 候補までは固定しない）
    func testCandidatesContainCommonKatakanaReadings() {
        let expected: [(String, String)] = [
            ("take", "ていく"), ("snow", "すのー"), ("time", "たいむ"), ("back", "ばっく"),
            ("game", "げーむ"), ("net", "ねっと"), ("test", "てすと"), ("happy", "はっぴー"),
            ("live", "らいぶ"), ("cute", "きゅーと"), ("home", "ほーむ"), ("night", "ないと"),
            ("music", "みゅーじっく"), ("song", "そんぐ"), ("pink", "ぴんく"), ("SnowMan", "すのーまん"),
        ]
        for (word, reading) in expected {
            let candidates = EnglishReading.phonics(word)
            XCTAssertTrue(candidates.contains(reading), "\(word) の候補に \(reading) がありません: \(candidates)")
        }
    }

    func testAcronymsAreSpelled() {
        XCTAssertEqual(EnglishReading.bestGuess("SNS"), "えすえぬえす")
        XCTAssertEqual(EnglishReading.bestGuess("OK"), "おーけー")
        XCTAssertEqual(EnglishReading.bestGuess("AI"), "えーあい")
        XCTAssertEqual(EnglishReading.bestGuess("LK"), "えるけー")
        XCTAssertEqual(EnglishReading.bestGuess("VTuber"), "ぶい" + EnglishReading.phonics("Tuber")[0])
    }

    func testNormalizeLatinJoinsWordsAndHalfwidths() {
        XCTAssertEqual(EnglishReading.normalizeLatin("Snow Manのライブ"), "SnowManのライブ")
        XCTAssertEqual(EnglishReading.normalizeLatin("ＢＴＳ と BTS"), "BTS と BTS")
        XCTAssertEqual(EnglishReading.latinRuns(in: "SNSでYouTubeを見る"), ["SNS", "YouTube"])
    }

    /// 語彙に無い英単語は 1 文字ずつのアルファベット読みではなく 1 語として読む
    func testUnknownLatinWordIsOneMorpheme() {
        let morphology = ViterbiMorphology()
        let morphemes = morphology.tokenize("take")
        XCTAssertEqual(morphemes.count, 1)
        XCTAssertEqual(morphemes.first?.reading, EnglishReading.bestGuess("take"))
    }

    /// 転写から英字の語を獲得する（読みは綴りの規則による推定）
    func testLexiconLearningAcquiresLatinWords() {
        let result = ViterbiMorphology.learnLexiconEntries(sentences: ["Snow Manのライブ", "SNSで見た"], existing: [])
        let learned = result.lexicon.filter { $0.cost == ViterbiMorphology.learnedLexiconCost }
        XCTAssertEqual(learned.first { $0.surface == "SnowMan" }?.reading, EnglishReading.bestGuess("SnowMan"))
        XCTAssertEqual(learned.first { $0.surface == "SNS" }?.reading, "えすえぬえす")
    }

    /// 外来語の拍（ティ、ウィ など）は 1 拍の音素 2 つになる
    func testForeignMorasAreSingleMoras() {
        let vocabulary = PhonemeVocabulary()
        XCTAssertEqual(vocabulary.kanaToPhonemes("てぃ"), ["t", "i"])
        XCTAssertEqual(vocabulary.kanaToPhonemes("うぃ"), ["w", "i"])
        XCTAssertEqual(vocabulary.kanaToMoras("ぱーてぃー").count, 4)
    }

    func testNormalizeLatinJoinsSymbolsInsideNames() {
        XCTAssertEqual(EnglishReading.normalizeLatin("K-POPとM!LKとBE：FIRST"), "KPOPとMLKとBEFIRST")
        XCTAssertEqual(EnglishReading.normalizeLatin("Tシャツ - 新作"), "Tシャツ - 新作")
    }

    /// コーパスが与える読みは綴りの規則より優先され、獲得済みの語の読みも置き換える
    func testLexiconLearningUsesGivenReadings() {
        let first = ViterbiMorphology.learnLexiconEntries(sentences: ["VIVANTを見た"], existing: [])
        XCTAssertEqual(first.lexicon.first { $0.surface == "VIVANT" }?.reading, EnglishReading.bestGuess("VIVANT"))
        let second = ViterbiMorphology.learnLexiconEntries(sentences: ["VIVANTを見た", "K-POPが好き"], existing: first.lexicon, readings: ["VIVANT": "ビバン", "KPOP": "ケーポップ"])
        XCTAssertEqual(second.lexicon.filter { $0.surface == "VIVANT" }.map { $0.reading }, ["びばん"])
        XCTAssertEqual(second.updated, 1, "新しい語が無くても既存の読みの置き換えは結果に含まれる")
        XCTAssertEqual(second.lexicon.first { $0.surface == "KPOP" }?.reading, "けーぽっぷ")
    }

    /// 語彙に無い外来語は後ろから 3 拍目（特殊拍なら 1 拍前）に核を置く
    func testLoanwordAccentKernel() {
        XCTAssertEqual(ViterbiMorphology.loanwordAccentKernel(reading: "すのーまん"), 2)
        XCTAssertEqual(ViterbiMorphology.loanwordAccentKernel(reading: "すぽーつ"), 2)
        XCTAssertEqual(ViterbiMorphology.loanwordAccentKernel(reading: "ぱーてぃー"), 1)
        XCTAssertEqual(ViterbiMorphology.loanwordAccentKernel(reading: "こっぷ"), 1)
        XCTAssertEqual(ViterbiMorphology.loanwordAccentKernel(reading: "ぱん"), 1)
    }

    /// 獲得した外来語は規則による核の推定値で登録され、漢字語は未知（0）のまま
    func testLearnedLoanwordsGetInitialKernel() {
        let result = ViterbiMorphology.learnLexiconEntries(sentences: ["コップに水を入れる", "SNSで見た"], existing: [])
        XCTAssertEqual(result.lexicon.first { $0.surface == "コップ" }?.accentKernel, 1)
        XCTAssertEqual(result.lexicon.first { $0.surface == "SNS" }?.accentKernel, ViterbiMorphology.loanwordAccentKernel(reading: "えすえぬえす"))
        XCTAssertEqual(result.lexicon.first { $0.surface == "水" }?.accentKernel, 0)
    }
}
