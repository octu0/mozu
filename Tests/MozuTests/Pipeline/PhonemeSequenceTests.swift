import XCTest
@testable import Mozu

final class PhonemeSequenceTests: XCTestCase {
    func testBoundarySilenceAndPauseInsertion() {
        let engine = MozuEngine()
        let seq = engine.sequenceBuilder.build(text: "こんにちは、本日は晴天なり。")
        XCTAssertEqual(seq.tokens.first?.id, PhonemeVocabulary.silId)
        XCTAssertEqual(seq.tokens.last?.id, PhonemeVocabulary.silId)
        XCTAssertTrue(seq.tokens.contains { $0.id == PhonemeVocabulary.pauId }, "読点の位置にポーズ音素が入る")
        XCTAssertEqual(seq.features.count, seq.tokens.count)
        for f in seq.features {
            XCTAssertEqual(f.count, PhonemeSequenceBuilder.featureCount)
        }
        // 無音・ポーズの特徴はすべて 0
        XCTAssertEqual(seq.features[0], [Float](repeating: 0, count: PhonemeSequenceBuilder.featureCount))
        // 句頭モーラのフラグが立つ音素がある
        XCTAssertTrue(seq.features.contains { $0[3] == 1.0 })
    }

    func testLabelVocabulary() {
        var labels = LabelVocabulary()
        labels.register(kind: "gender", value: "female")
        labels.register(kind: "gender", value: "male")
        labels.register(kind: "gender", value: "female")
        XCTAssertEqual(labels.count(kind: "gender"), 3)
        XCTAssertEqual(labels.index(kind: "gender", value: "male"), 2)
        XCTAssertEqual(labels.index(kind: "gender", value: "unknown"), 0)
        XCTAssertEqual(labels.index(kind: "age", value: "10代"), 0)
        XCTAssertEqual(labels.indices(for: ["gender": "female"]), [1])
        XCTAssertEqual(labels.value(kind: "gender", index: 2), "male")
    }
}
