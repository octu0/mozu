import XCTest
import MLX
@testable import Mozu

/// 録音との比較による採点と、評価セットの選び方を検証する
final class ProsodyEvaluationTests: XCTestCase {
    /// 合成結果そのものを「録音」にすると、比は 1、誤差は 0 になる
    func testScoreOfOwnOutputIsPerfect() throws {
        let config = AcousticDefaults.config(labelKinds: [], modelDim: 16, encoderLayers: 1, decoderLayers: 1)
        let model = AcousticModel(weights: MLXAcousticModel(config: config, labelCounts: []).exportWeights())
        let sequence = MozuEngine().sequenceBuilder.build(text: "今日は、いい天気ですね。")
        let out = model.synthesize(phonemeIds: sequence.ids, features: sequence.features, labelIndices: [])
        let reference = UtteranceSample(id: "x", text: "", phonemeIds: sequence.ids.map { Int32($0) }, features: sequence.features, labelIndices: [], durations: out.durations, pitch: out.pitch, voiced: sequence.ids.map { _ in true }, voicedFrames: out.durations, energy: out.energy, mel: out.mel, pcm: [])
        let score = try XCTUnwrap(ProsodyEvaluation.score(model: model, reference: reference, sequence: sequence, labelIndices: []))
        XCTAssertEqual(score.durationRatio, 1, accuracy: 1e-5)
        XCTAssertEqual(score.durationLogError, 0, accuracy: 1e-5)
        XCTAssertEqual(score.pitchRangeRatio, 1, accuracy: 1e-3)
        XCTAssertEqual(score.pitchContourError, 0, accuracy: 1e-3)
        XCTAssertEqual(score.pitchNormalizedError, 0, accuracy: 1e-3)
        XCTAssertEqual(score.energyNormalizedError, 0, accuracy: 1e-3)
        XCTAssertEqual(score.energyRangeRatio, 1, accuracy: 1e-3)
        XCTAssertEqual(score.melDistance, 0, accuracy: 1e-4)
        XCTAssertEqual(score.melContrastRatio, 1, accuracy: 1e-3)
        XCTAssertEqual(score.melFluxRatio, 1, accuracy: 1e-3)
        XCTAssertEqual(score.composite, 0, accuracy: 1e-2)

        // 音素列が違えば採点しない
        let other = MozuEngine().sequenceBuilder.build(text: "雨です。")
        XCTAssertNil(ProsodyEvaluation.score(model: model, reference: reference, sequence: other, labelIndices: []))
    }

    /// 比の平均は対数で取る（2 倍と半分の平均は 1）
    func testAverageUsesGeometricMeanForRatios() throws {
        func score(_ ratio: Float) -> ProsodyScore {
            return ProsodyScore(id: "", durationRatio: ratio, durationLogError: 0, pauseShare: 0, referencePauseShare: 0, pitchRangeRatio: ratio, pitchContourError: 0, pitchCorrelation: 0, pitchLevelDifference: 0, pitchNormalizedError: 0, energyRangeRatio: 1, energyJumpRatio: 1, energyContourError: 0, energyNormalizedError: 0, melDistance: 0, melContrastRatio: 1, melFluxRatio: 1)
        }
        let avg = try XCTUnwrap(ProsodyEvaluation.average([score(2), score(0.5)]))
        XCTAssertEqual(avg.durationRatio, 1, accuracy: 1e-5)
        XCTAssertEqual(avg.pitchRangeRatio, 1, accuracy: 1e-5)
    }

    /// 小さいコーパスにも最低枠を配り、話者を順に回して選ぶ。同じ候補からは同じ集合になる
    func testSelectionSpreadsAcrossCorporaAndSpeakers() {
        var candidates: [(corpus: String, id: String, speaker: String)] = []
        for i in 0..<1000 {
            candidates.append(("big", "b\(i)", "s\(i % 50)"))
        }
        for i in 0..<40 {
            candidates.append(("small", "k\(i)", "one"))
        }
        let set = EvaluationSet.select(candidates: candidates, count: 100, minPerCorpus: 20)
        XCTAssertEqual(set.entries.count, 100)
        XCTAssertEqual(set.entries.filter { $0.corpus == "small" }.count, 20)
        let big = set.entries.filter { $0.corpus == "big" }.map { Int($0.id.dropFirst())! }
        XCTAssertEqual(Set(big.map { $0 % 50 }).count, 50, "大きいコーパスでは全話者から選ぶ")
        XCTAssertEqual(Set(set.entries.map { $0.id }).count, 100, "重複しない")
        XCTAssertEqual(EvaluationSet.select(candidates: candidates, count: 100, minPerCorpus: 20).entries, set.entries)
    }
}
