import XCTest
import Foundation
import MLX
@testable import Mozu

/// 音響モデルの MLX 順伝播（教師強制）と CPU 推論（継続時間を与えた場合）の一致を検証する
final class AcousticParityTests: XCTestCase {
    func testCPUMatchesMLX() {
        let labels = LabelVocabulary(kinds: ["gender"], values: ["gender": ["female", "male"]])
        var config = AcousticDefaults.config(labelKinds: labels.kinds, modelDim: 32, encoderLayers: 1, decoderLayers: 1)
        config = config.withNormalization(pitchMean: 0.1, pitchStd: 0.5, energyMean: -3.0, energyStd: 2.0)
        let model = MLXAcousticModel(config: config, labelCounts: [labels.count(kind: "gender")])
        // ラベル埋め込みを 0 以外にしておく
        model.labelEmbeddings[0] = MLXRandom.uniform(low: -0.5, high: 0.5, model.labelEmbeddings[0].shape)
        let weights = model.exportWeights()
        let cpu = AcousticModel(weights: weights)

        let ids: [Int32] = [1, 5, 6, 10, 7, 1]
        let features: [[Float]] = ids.map { _ in [1, 0.5, 0, 1, 0, 0] }
        let durations = [3, 4, 2, 5, 3, 2]
        let pitch: [Float] = [0, 0.3, -0.2, 0, 0.1, 0]
        let voiced = [false, true, true, false, true, false]
        let energy: [Float] = [-6, -2, -1, -3, -2, -6]
        let frameCount = durations.reduce(0, +)
        let mel = [[Float]](repeating: [Float](repeating: -3, count: config.melChannels), count: frameCount)
        let sample = UtteranceSample(id: "x", text: "", phonemeIds: ids, features: features, labelIndices: [2], durations: durations, pitch: pitch, voiced: voiced, voicedFrames: [0, 3, 2, 0, 3, 0], energy: energy, mel: mel, pcm: [])
        let trainer = MLXAcousticTrainer(model: model, learningRate: 1e-4)
        var rng = SystemRandomNumberGenerator()
        let batch = trainer.makeBatch([sample], dropLabels: false, rng: &rng)
        let pred = model.forward(batch)
        eval(pred.mel, pred.logDurations, pred.pitch, pred.energy)
        let mlxMel = pred.mel.asArray(Float.self)
        let mlxPitch = pred.pitch.asArray(Float.self)

        // CPU: 教師強制に合わせるため、ピッチ・エネルギーの予測値ではなく教師値を使いたいが、
        // CPU 推論は予測値を使う。そこでヘッドの出力が教師値と一致する入力ではなく、両者が同じ入力（予測値）を使う形で比較する。
        // CPU の出力 Mel を MLX と比べるには、MLX 側にも予測値を条件として与える必要があるので、
        // ここでは教師値 = 予測値 となるサンプルを作って比較する。
        let predictedPitch = mlxPitch.map { $0 * config.pitchStd + config.pitchMean }
        let predictedEnergy = pred.energy.asArray(Float.self).map { $0 * config.energyStd + config.energyMean }
        let consistent = UtteranceSample(id: "x", text: "", phonemeIds: ids, features: features, labelIndices: [2], durations: durations, pitch: predictedPitch, voiced: [true, true, true, true, true, true], voicedFrames: [3, 3, 3, 3, 3, 3], energy: predictedEnergy, mel: mel, pcm: [])
        let batch2 = trainer.makeBatch([consistent], dropLabels: false, rng: &rng)
        let pred2 = model.forward(batch2)
        eval(pred2.mel)
        let mlxMel2 = pred2.mel.asArray(Float.self)
        _ = mlxMel

        let out = cpu.synthesize(phonemeIds: ids.map { Int($0) }, features: features, labelIndices: [2], durations: durations)
        XCTAssertEqual(out.mel.count, frameCount)
        var t = 0
        while t < frameCount {
            var c = 0
            while c < config.melChannels {
                XCTAssertEqual(out.mel[t][c], mlxMel2[t * config.melChannels + c], accuracy: 3e-3, "frame \(t) ch \(c)")
                c += 1
            }
            t += 1
        }
        var p = 0
        while p < ids.count {
            XCTAssertEqual(out.pitch[p], predictedPitch[p], accuracy: 3e-3)
            p += 1
        }
    }
}
