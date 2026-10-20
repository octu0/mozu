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

    /// 話し方の潜在変数つき: 参照エンコーダの平均と、その z を与えた合成が CPU と MLX で一致する
    func testCPUMatchesMLXWithStyle() throws {
        let labels = LabelVocabulary(kinds: ["gender"], values: ["gender": ["female", "male"]])
        var config = AcousticDefaults.config(labelKinds: labels.kinds, modelDim: 32, encoderLayers: 1, decoderLayers: 1)
        config = config.withNormalization(pitchMean: 0.1, pitchStd: 0.5, energyMean: -3.0, energyStd: 2.0)
        let model = MLXAcousticModel(config: config, labelCounts: [labels.count(kind: "gender")])
        let style = MLXStyleEncoder(melChannels: config.melChannels, modelDim: config.modelDim, latentDim: 4, refDim: 16, layers: 1)
        style.styleProjection = MLXRandom.uniform(low: -0.5, high: 0.5, style.styleProjection.shape)
        model.style = style
        let weights = model.exportWeights()
        let cpu = AcousticModel(weights: weights)

        let ids: [Int32] = [1, 5, 6, 10, 7, 1]
        let features: [[Float]] = ids.map { _ in [1, 0.5, 0, 1, 0, 0] }
        let durations = [3, 4, 2, 5, 3, 2]
        let frameCount = durations.reduce(0, +)
        let mel = (0..<frameCount).map { t in (0..<config.melChannels).map { c in -4.0 + Float((t * 7 + c * 3) % 11) * 0.3 } }
        let trainer = MLXAcousticTrainer(model: model, learningRate: 1e-4)
        var rng = SystemRandomNumberGenerator()
        func sample(pitch: [Float], energy: [Float]) -> UtteranceSample {
            return UtteranceSample(id: "x", text: "", phonemeIds: ids, features: features, labelIndices: [1], durations: durations, pitch: pitch, voiced: [true, true, true, true, true, true], voicedFrames: [3, 3, 3, 3, 3, 3], energy: energy, mel: mel, pcm: [])
        }
        // 参照エンコーダの平均
        let first = model.forward(trainer.makeBatch([sample(pitch: [0, 0, 0, 0, 0, 0], energy: [-3, -3, -3, -3, -3, -3])], dropLabels: false, latent: .posterior, rng: &rng))
        eval(first.pitch, first.energy, first.latentMean!)
        let mlxMean = first.latentMean!.asArray(Float.self)
        let cpuMean = StyleEncoder(weights: weights.style!).encode(mel: mel).mean
        for (a, b) in zip(cpuMean, mlxMean) {
            XCTAssertEqual(a, b, accuracy: 2e-3, "参照エンコーダの平均が一致しません")
        }
        // 予測ピッチ・エネルギーを教師に入れ直して、CPU（予測値を使う）と同じ条件で比べる
        let predictedPitch = first.pitch.asArray(Float.self).map { $0 * config.pitchStd + config.pitchMean }
        let predictedEnergy = first.energy.asArray(Float.self).map { $0 * config.energyStd + config.energyMean }
        let second = model.forward(trainer.makeBatch([sample(pitch: predictedPitch, energy: predictedEnergy)], dropLabels: false, latent: .posterior, rng: &rng))
        eval(second.mel, second.logDurations)
        let mlxMel = second.mel.asArray(Float.self)
        let out = cpu.synthesize(phonemeIds: ids.map { Int($0) }, features: features, labelIndices: [1], latent: cpuMean, durations: durations)
        var maxDiff: Float = 0
        for t in 0..<frameCount {
            for c in 0..<config.melChannels {
                maxDiff = max(maxDiff, abs(out.mel[t][c] - mlxMel[t * config.melChannels + c]))
            }
        }
        XCTAssertLessThan(maxDiff, 3e-3, "潜在変数つきの Mel が一致しません")
        // 継続時間ヘッドにも z が入っている
        let free = cpu.synthesize(phonemeIds: ids.map { Int($0) }, features: features, labelIndices: [1], latent: cpuMean)
        let mlxDur = second.logDurations.asArray(Float.self).map { max(1, Int(expf($0).rounded())) }
        XCTAssertEqual(Array(free.durations.dropFirst().dropLast()), Array(mlxDur.dropFirst().dropLast()))
        // z = 0 と z なしは同じ
        let zero = cpu.synthesize(phonemeIds: ids.map { Int($0) }, features: features, labelIndices: [1], latent: [0, 0, 0, 0], durations: durations)
        let none = cpu.synthesize(phonemeIds: ids.map { Int($0) }, features: features, labelIndices: [1], durations: durations)
        XCTAssertEqual(zero.mel, none.mel)
    }

    /// 話し方を持たない重み（以前の形式）もそのまま読める
    func testWeightsWithoutStyleStillDecode() throws {
        let labels = LabelVocabulary(kinds: ["gender"], values: ["gender": ["female"]])
        let config = AcousticDefaults.config(labelKinds: labels.kinds, modelDim: 16, encoderLayers: 1, decoderLayers: 1)
        let weights = MLXAcousticModel(config: config, labelCounts: [2]).exportWeights()
        XCTAssertNil(weights.style)
        let data = try JSONEncoder().encode(weights)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("\"style\""))
        let decoded = try JSONDecoder().decode(AcousticWeights.self, from: data)
        XCTAssertNil(decoded.style)
        XCTAssertEqual(decoded, weights)
    }
}
