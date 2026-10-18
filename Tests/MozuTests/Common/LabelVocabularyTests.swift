import XCTest
@testable import Mozu

/// 声の指定（ラベルの値の並び）の解決を検証する
final class LabelVocabularyTests: XCTestCase {
    private func vocabulary() -> LabelVocabulary {
        var v = LabelVocabulary()
        v.register(kind: "gender", value: "female")
        v.register(kind: "gender", value: "male")
        v.register(kind: "age", value: "10代後半")
        v.register(kind: "dialect", value: "標準語")
        v.register(kind: "character", value: "ギャル")
        v.register(kind: "register", value: "アニメ調")
        v.register(kind: "pitch", value: "高め")
        v.register(kind: "timbre", value: "高め")
        return v
    }

    func testBareValuesResolveToTheirKinds() throws {
        let labels = try vocabulary().resolve("Female, 標準語,ギャル,アニメ調,age=10代後半")
        XCTAssertEqual(labels, ["gender": "female", "dialect": "標準語", "character": "ギャル", "register": "アニメ調", "age": "10代後半"])
    }

    func testAmbiguousAndUnknownValuesAreErrors() {
        XCTAssertThrowsError(try vocabulary().resolve("高め")) { error in
            XCTAssertEqual(error as? LabelSpecError, .ambiguous("高め", kinds: ["pitch", "timbre"]))
        }
        XCTAssertEqual(try vocabulary().resolve("pitch=高め"), ["pitch": "高め"])
        XCTAssertThrowsError(try vocabulary().resolve("ギャル語")) { error in
            XCTAssertEqual(error as? LabelSpecError, .unknownValue("ギャル語", suggestions: ["character=ギャル"]))
        }
        XCTAssertThrowsError(try vocabulary().resolve("style=アニメ調"))
    }
}
