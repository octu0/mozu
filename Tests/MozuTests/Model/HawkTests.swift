import XCTest
import Foundation
import MLX
@testable import Mozu

/// Hawk 幹の MLX 学習側と CPU 推論側の一致、および詰め物のマスクを検証する
final class HawkTests: XCTestCase {
    private func randomInput(frames: Int, dim: Int, seed: UInt64) -> [Float] {
        var state = seed
        var out = [Float](repeating: 0.0, count: frames * dim)
        var i = 0
        while i < out.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            out[i] = Float(Int64(bitPattern: state >> 11) % 2000) / 1000.0 - 1.0
            i += 1
        }
        return out
    }

    /// 長さの違う 2 系列をまとめて MLX で処理した結果が、CPU で 1 系列ずつ処理した結果と一致する。
    /// 短い方の系列が一致すれば、双方向走査が詰め物の影響を受けていないことになる。
    func testCPUMatchesMLXWithPadding() {
        let config = HawkStackConfig(modelDim: 32, rnnDim: 32, mlpDim: 64, heads: 2, convLeft: 2, convRight: 2, numLayers: 2, bidirectional: true)
        let stack = MLXHawkStack(config: config)
        let weights = stack.exportWeights()
        let cpu = HawkStack(weights: weights)

        let framesA = 9
        let framesB = 5
        let dim = config.modelDim
        let xA = randomInput(frames: framesA, dim: dim, seed: 1)
        let xB = randomInput(frames: framesB, dim: dim, seed: 2)
        var batch = xA
        batch.append(contentsOf: xB)
        batch.append(contentsOf: [Float](repeating: 0.0, count: (framesA - framesB) * dim))
        let input = MLXArray(batch, [2, framesA, dim])
        let mask = makeFrameMask(lengths: [framesA, framesB], frames: framesA)
        let output = stack(input, frameMask: mask)
        eval(output)
        let flat = output.asArray(Float.self)

        let cpuA = cpu.forward(xA, frames: framesA)
        let cpuB = cpu.forward(xB, frames: framesB)
        var i = 0
        while i < framesA * dim {
            XCTAssertEqual(flat[i], cpuA[i], accuracy: 2e-3, "系列 A の要素 \(i) が一致しません")
            i += 1
        }
        i = 0
        while i < framesB * dim {
            XCTAssertEqual(flat[framesA * dim + i], cpuB[i], accuracy: 2e-3, "系列 B（詰め物あり）の要素 \(i) が一致しません")
            i += 1
        }
    }

    /// 重みの JSON 往復
    func testWeightsRoundTrip() throws {
        let config = HawkStackConfig(modelDim: 16, rnnDim: 16, mlpDim: 32, heads: 2, convLeft: 1, convRight: 1, numLayers: 1, bidirectional: true)
        let weights = MLXHawkStack(config: config).exportWeights()
        let data = try JSONEncoder().encode(weights)
        let decoded = try JSONDecoder().decode(HawkStackWeights.self, from: data)
        XCTAssertEqual(decoded, weights)
    }
}
