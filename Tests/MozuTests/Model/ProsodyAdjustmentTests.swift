import XCTest
@testable import Mozu

/// 合成時の抑揚・エネルギー調整を検証する
final class ProsodyAdjustmentTests: XCTestCase {
    func testIntonationScalesAroundMeanOfSpokenPhonemes() {
        let ids = [PhonemeVocabulary.silId, 5, 6, PhonemeVocabulary.pauId, 7]
        var pitch: [Float] = [-3.0, 1.0, -1.0, 2.0, 0.0]
        var energy: [Float] = [0, 0, 0, 0, 0]
        AcousticModel.adjustProsody(pitchZ: &pitch, energyZ: &energy, phonemeIds: ids, intonation: 2.0, energyShift: 0.5)
        // 無音・ポーズを除く平均は 0。そこからのずれが 2 倍になる（±3.5 で頭打ち）
        XCTAssertEqual(pitch, [-3.5, 2.0, -2.0, 3.5, 0.0])
        XCTAssertEqual(energy, [0.5, 0.5, 0.5, 0.5, 0.5])
    }

    func testDefaultsLeaveProsodyUnchanged() {
        var pitch: [Float] = [0.3, -0.2]
        var energy: [Float] = [0.1, 0.4]
        AcousticModel.adjustProsody(pitchZ: &pitch, energyZ: &energy, phonemeIds: [5, 6], intonation: 1.0, energyShift: 0.0)
        XCTAssertEqual(pitch, [0.3, -0.2])
        XCTAssertEqual(energy, [0.1, 0.4])
    }
}
