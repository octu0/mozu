#if canImport(MLX)
import Foundation
import MLX
import MLXNN

/// 線形再帰 h_t = a_t h_{t−1} + x_t（h_{−1} = 0）を [B, T, R] に対して計算する Metal カーネル
///
/// 逆伝播も逆時間走査のカーネルなので、グラフの大きさが系列長に比例して伸びない。
public enum MLXLinearScan {
    static let forwardKernel = MLXFast.metalKernel(
        name: "mozu_linear_scan_forward",
        inputNames: ["a", "x"],
        outputNames: ["h"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float state = 0.0f;
                for (int t = 0; t < T; ++t) {
                    size_t idx = base + size_t(t) * R;
                    state = a[idx] * state + x[idx];
                    h[idx] = state;
                }
            }
            """
    )

    /// δ_t = g_t + a_{t+1} δ_{t+1}, dx_t = δ_t, da_t = δ_t h_{t−1}
    static let backwardKernel = MLXFast.metalKernel(
        name: "mozu_linear_scan_backward",
        inputNames: ["a", "h", "g"],
        outputNames: ["da", "dx"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float carry = 0.0f;
                for (int t = T - 1; 0 <= t; --t) {
                    size_t idx = base + size_t(t) * R;
                    float d = g[idx] + carry;
                    dx[idx] = d;
                    float previous = 0.0f;
                    if (0 < t) {
                        previous = h[idx - R];
                    }
                    da[idx] = d * previous;
                    carry = d * a[idx];
                }
            }
            """
    )

    static func launch(_ shape: [Int]) -> (template: [(String, any KernelTemplateArg)], grid: (Int, Int, Int), group: (Int, Int, Int)) {
        let threads = shape[0] * shape[2]
        return (
            template: [("B", shape[0]), ("T", shape[1]), ("R", shape[2])],
            grid: (threads, 1, 1),
            group: (min(256, threads), 1, 1)
        )
    }

    static func forward(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        let p = launch(a.shape)
        return forwardKernel(
            [a, x], template: p.template, grid: p.grid, threadGroup: p.group,
            outputShapes: [a.shape], outputDTypes: [.float32]
        )[0]
    }

    nonisolated(unsafe) static let function = CustomFunction {
        Forward { inputs in
            return [MLXLinearScan.forward(inputs[0], inputs[1])]
        }
        VJP { primals, cotangents in
            let a = primals[0]
            let h = MLXLinearScan.forward(a, primals[1])
            let p = launch(a.shape)
            return backwardKernel(
                [a, h, cotangents[0]], template: p.template, grid: p.grid, threadGroup: p.group,
                outputShapes: [a.shape, a.shape], outputDTypes: [.float32, .float32]
            )
        }
    }

    /// a, x: [B, T, R]（float32）→ h: [B, T, R]
    public static func scan(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        return function([a.asType(.float32), x.asType(.float32)])[0]
    }
}

/// GeLU の近似 x σ(1.702 x)。CPU 推論と同じ式
@inline(__always)
public func hawkGelu(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(1.702 * x)
}

@inline(__always)
public func hawkRMSNorm(_ x: MLXArray, _ gain: MLXArray) -> MLXArray {
    return x * rsqrt(mean(x * x, axis: -1, keepDims: true) + HawkConstants.rmsNormEpsilon) * gain
}

/// 時間軸（axis 1）の反転
func reverseTime(_ x: MLXArray) -> MLXArray {
    let frames = x.shape[1]
    let indices = MLXArray((0..<frames).reversed().map { Int32($0) })
    return take(x, indices, axis: 1)
}

/// Hawk 1 層（MLX）。重みの意味は `HawkLayerWeights` を参照
public final class MLXHawkLayer: Module, @unchecked Sendable {
    public var norm1: MLXArray
    public var wGate: MLXArray
    public var wX: MLXArray
    public var convW: MLXArray
    public var convB: MLXArray
    public var wA: MLXArray
    public var bA: MLXArray
    public var wI: MLXArray
    public var bI: MLXArray
    public var lambda: MLXArray
    public var wO: MLXArray
    public var norm2: MLXArray
    public var wMLPGate: MLXArray
    public var wMLPUp: MLXArray
    public var wMLPDown: MLXArray

    public let config: HawkStackConfig

    public init(config: HawkStackConfig) {
        self.config = config
        let modelDim = config.modelDim
        let rnnDim = config.rnnDim
        let mlpDim = config.mlpDim
        let headDim = config.headDim
        func uniform(_ fanIn: Int, _ shape: [Int], scale: Float = 1.0) -> MLXArray {
            let bound = scale * sqrt(3.0 / Float(fanIn))
            return MLXRandom.uniform(low: -bound, high: bound, shape)
        }
        // 残差への射影は 1/√(2L) で縮め、初期の残差分散が深さに依らないようにする
        let residualScale = 1.0 / sqrt(Float(2 * config.numLayers))
        self.norm1 = MLXArray.ones([modelDim])
        self.wGate = uniform(modelDim, [modelDim, rnnDim])
        self.wX = uniform(modelDim, [modelDim, rnnDim])
        self.convW = uniform(config.convTaps, [config.convTaps, rnnDim])
        self.convB = MLXArray.zeros([rnnDim])
        self.wA = uniform(headDim, [config.heads, headDim, headDim])
        self.bA = MLXArray.zeros([rnnDim])
        self.wI = uniform(headDim, [config.heads, headDim, headDim])
        self.bI = MLXArray.zeros([rnnDim])
        self.lambda = MLXArray(HawkConstants.initialLambda(rnnDim: rnnDim), [rnnDim])
        self.wO = uniform(rnnDim, [rnnDim, modelDim], scale: residualScale)
        self.norm2 = MLXArray.ones([modelDim])
        self.wMLPGate = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPUp = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPDown = uniform(mlpDim, [mlpDim, modelDim], scale: residualScale)
        super.init()
    }

    public convenience init(config: HawkStackConfig, weights: HawkLayerWeights) {
        self.init(config: config)
        load(weights)
    }

    public func load(_ w: HawkLayerWeights) {
        norm1 = MLXArray(w.norm1.values, w.norm1.shape)
        wGate = MLXArray(w.wGate.values, w.wGate.shape)
        wX = MLXArray(w.wX.values, w.wX.shape)
        convW = MLXArray(w.convW.values, w.convW.shape)
        convB = MLXArray(w.convB.values, w.convB.shape)
        wA = MLXArray(w.wA.values, w.wA.shape)
        bA = MLXArray(w.bA.values, w.bA.shape)
        wI = MLXArray(w.wI.values, w.wI.shape)
        bI = MLXArray(w.bI.values, w.bI.shape)
        lambda = MLXArray(w.lambda.values, w.lambda.shape)
        wO = MLXArray(w.wO.values, w.wO.shape)
        norm2 = MLXArray(w.norm2.values, w.norm2.shape)
        wMLPGate = MLXArray(w.wMLPGate.values, w.wMLPGate.shape)
        wMLPUp = MLXArray(w.wMLPUp.values, w.wMLPUp.shape)
        wMLPDown = MLXArray(w.wMLPDown.values, w.wMLPDown.shape)
    }

    public func exportWeights() -> HawkLayerWeights {
        return HawkLayerWeights(
            norm1: tensor(norm1), wGate: tensor(wGate), wX: tensor(wX), convW: tensor(convW), convB: tensor(convB),
            wA: tensor(wA), bA: tensor(bA), wI: tensor(wI), bI: tensor(bI), lambda: tensor(lambda), wO: tensor(wO),
            norm2: tensor(norm2), wMLPGate: tensor(wMLPGate), wMLPUp: tensor(wMLPUp), wMLPDown: tensor(wMLPDown)
        )
    }

    /// x: [B, T, modelDim] → [B, T, modelDim]。`frameMask` [B, T, 1] は各系列の終端より後ろを 0 にする。
    /// なぜマスクが要るか: 逆方向の走査は末尾の詰め物から本体へ向かって走るので、詰め物の入力を 0 にしないと
    /// 詰め物の状態が本体へ漏れる。畳み込みも終端の向こうを見るため、畳み込み入力も 0 にする。
    public func callAsFunction(_ x: MLXArray, frameMask: MLXArray?) -> MLXArray {
        let batch = x.shape[0]
        let frames = x.shape[1]
        let rnnDim = config.rnnDim
        let headDim = config.headDim

        let n = hawkRMSNorm(x, norm1)
        let branch = hawkGelu(matmul(n, wGate))
        var raw = matmul(n, wX)
        if let mask = frameMask {
            raw = raw * mask
        }
        let paddedRaw = concatenated(
            [MLXArray.zeros([batch, config.convLeft, rnnDim]), raw, MLXArray.zeros([batch, config.convRight, rnnDim])], axis: 1)
        var u = convB + MLXArray.zeros([batch, frames, rnnDim])
        var k = 0
        while k < config.convTaps {
            u = u + paddedRaw[0..., k..<(k + frames), 0...] * convW[k]
            k += 1
        }
        let blocks = u.reshaped([batch, frames, config.heads, 1, headDim])
        let r = sigmoid(matmul(blocks, wA).reshaped([batch, frames, rnnDim]) + bA)
        let i = sigmoid(matmul(blocks, wI).reshaped([batch, frames, rnnDim]) + bI)
        // log a_t = c · r_t · log σ(Λ)
        let logA = (HawkConstants.decayExponent * r) * logSigmoid(lambda)
        let a = exp(logA)
        var input = sqrt(1.0 - exp(2.0 * logA) + 1e-6) * (i * u)
        if let mask = frameMask {
            input = input * mask
        }
        let h: MLXArray
        if config.bidirectional {
            let half = rnnDim / 2
            let hForward = MLXLinearScan.scan(a[0..., 0..., 0..<half], input[0..., 0..., 0..<half])
            let hBackward = reverseTime(MLXLinearScan.scan(
                reverseTime(a[0..., 0..., half...]), reverseTime(input[0..., 0..., half...])))
            h = concatenated([hForward, hBackward], axis: -1)
        } else {
            h = MLXLinearScan.scan(a, input)
        }
        let mixed = x + matmul(branch * h, wO)

        let m = hawkRMSNorm(mixed, norm2)
        return mixed + matmul(hawkGelu(matmul(m, wMLPGate)) * matmul(m, wMLPUp), wMLPDown)
    }
}

/// Hawk 幹（MLX）。入出力射影は使う側が持つ
public final class MLXHawkStack: Module, @unchecked Sendable {
    public let config: HawkStackConfig
    public var layers: [MLXHawkLayer]

    public init(config: HawkStackConfig) {
        self.config = config
        var built: [MLXHawkLayer] = []
        var l = 0
        while l < config.numLayers {
            built.append(MLXHawkLayer(config: config))
            l += 1
        }
        self.layers = built
        super.init()
    }

    public convenience init(weights: HawkStackWeights) {
        self.init(config: weights.config)
        var l = 0
        while l < weights.layers.count {
            layers[l].load(weights.layers[l])
            l += 1
        }
    }

    public func callAsFunction(_ x: MLXArray, frameMask: MLXArray?) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, frameMask: frameMask)
        }
        return h
    }

    public func exportWeights() -> HawkStackWeights {
        return HawkStackWeights(config: config, layers: layers.map { $0.exportWeights() })
    }
}

/// MLXArray → Tensor（評価してホストへコピー）
public func tensor(_ x: MLXArray) -> Tensor {
    eval(x)
    return Tensor(shape: x.shape, values: x.asArray(Float.self))
}

/// 系列長から [B, T, 1] の float マスクを作る
public func makeFrameMask(lengths: [Int], frames: Int) -> MLXArray {
    let positions = MLXArray(0..<frames).reshaped([1, frames, 1])
    let counts = MLXArray(lengths.map { Int32($0) }).reshaped([-1, 1, 1])
    return (positions .< counts).asType(.float32)
}
#endif
