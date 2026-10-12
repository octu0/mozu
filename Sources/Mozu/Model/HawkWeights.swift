import Foundation

/// Hawk 幹（Griffin の再帰ブロックのみの構成、De et al. 2024）の設定
///
/// 1 層は「RMSNorm → 再帰ブロック → 残差」「RMSNorm → ゲート付き MLP → 残差」。
/// 再帰ブロックは GeLU(x·Wg) の枝と、x·Wx に時間畳み込みと RG-LRU を掛けた枝の積を Wo で射影する。
/// 音声合成は文全体を見てよいので、再帰は双方向にできる（`bidirectional`）。
/// 双方向のとき再帰幅の前半は順方向、後半は逆方向に走査する。
public struct HawkStackConfig: Sendable, Codable, Equatable {
    public let modelDim: Int
    public let rnnDim: Int
    public let mlpDim: Int
    /// RG-LRU ゲート射影のブロック対角数
    public let heads: Int
    /// 畳み込みが見る過去フレーム数（現在フレームを含まない）
    public let convLeft: Int
    /// 畳み込みが見る未来フレーム数
    public let convRight: Int
    public let numLayers: Int
    public let bidirectional: Bool

    public init(modelDim: Int, rnnDim: Int, mlpDim: Int, heads: Int, convLeft: Int, convRight: Int, numLayers: Int, bidirectional: Bool) {
        precondition(modelDim % 8 == 0 && mlpDim % 8 == 0, "widths must be multiples of 8")
        precondition(rnnDim % heads == 0, "rnnDim must be divisible by heads")
        precondition(bidirectional != true || rnnDim % 16 == 0, "bidirectional rnnDim must be a multiple of 16")
        precondition(rnnDim % 8 == 0, "rnnDim must be a multiple of 8")
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convLeft = convLeft
        self.convRight = convRight
        self.numLayers = numLayers
        self.bidirectional = bidirectional
    }

    public var headDim: Int {
        return rnnDim / heads
    }

    public var convTaps: Int {
        return convLeft + 1 + convRight
    }
}

/// Hawk 1 層の重み。行列はすべて行優先 [入力, 出力]（x · W で適用）
public struct HawkLayerWeights: Sendable, Codable, Equatable {
    public var norm1: Tensor       // [modelDim]
    public var wGate: Tensor       // [modelDim, rnnDim] GeLU 枝
    public var wX: Tensor          // [modelDim, rnnDim] 再帰枝
    public var convW: Tensor       // [convTaps, rnnDim]（行 convLeft が現在フレーム）
    public var convB: Tensor       // [rnnDim]
    public var wA: Tensor          // [heads, headDim, headDim] 保持ゲート（ブロック対角）
    public var bA: Tensor          // [rnnDim]
    public var wI: Tensor          // [heads, headDim, headDim] 入力ゲート（ブロック対角）
    public var bI: Tensor          // [rnnDim]
    public var lambda: Tensor      // [rnnDim] 減衰 σ(Λ) のロジット
    public var wO: Tensor          // [rnnDim, modelDim]
    public var norm2: Tensor       // [modelDim]
    public var wMLPGate: Tensor    // [modelDim, mlpDim]
    public var wMLPUp: Tensor      // [modelDim, mlpDim]
    public var wMLPDown: Tensor    // [mlpDim, modelDim]

    public init(norm1: Tensor, wGate: Tensor, wX: Tensor, convW: Tensor, convB: Tensor, wA: Tensor, bA: Tensor, wI: Tensor, bI: Tensor, lambda: Tensor, wO: Tensor, norm2: Tensor, wMLPGate: Tensor, wMLPUp: Tensor, wMLPDown: Tensor) {
        self.norm1 = norm1
        self.wGate = wGate
        self.wX = wX
        self.convW = convW
        self.convB = convB
        self.wA = wA
        self.bA = bA
        self.wI = wI
        self.bI = bI
        self.lambda = lambda
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wGate.count + wX.count + convW.count + convB.count + wA.count + bA.count
            + wI.count + bI.count + lambda.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }
}

/// Hawk 幹の重み（入出力射影は持たない。幹を使う側が持つ）
public struct HawkStackWeights: Sendable, Codable, Equatable {
    public let config: HawkStackConfig
    public var layers: [HawkLayerWeights]

    public init(config: HawkStackConfig, layers: [HawkLayerWeights]) {
        precondition(layers.count == config.numLayers)
        self.config = config
        self.layers = layers
    }

    public var parameterCount: Int {
        return layers.reduce(0) { $0 + $1.parameterCount }
    }
}

public enum HawkConstants {
    /// RG-LRU の減衰指数 c: a_t = σ(Λ)^(c·r_t)
    public static let decayExponent: Float = 8.0
    /// RMSNorm の ε（MLX 学習と CPU 推論で共通）
    public static let rmsNormEpsilon: Float = 1e-5
    /// 初期時定数（フレーム）の範囲。r = 0.5 のときチャネルごとに対数等間隔に配る
    public static let initialTimeConstants: ClosedRange<Float> = 1.0...100.0

    /// σ(Λ)^(c/2) = exp(−1/τ) となる Λ の初期値
    public static func initialLambda(rnnDim: Int) -> [Float] {
        let lo = logf(initialTimeConstants.lowerBound)
        let hi = logf(initialTimeConstants.upperBound)
        var values = [Float](repeating: 0.0, count: rnnDim)
        var n = 0
        while n < rnnDim {
            var frac: Float = 0.0
            if 1 < rnnDim {
                frac = Float(n) / Float(rnnDim - 1)
            }
            let tau = expf(lo + (hi - lo) * frac)
            let a = expf(-1.0 / (tau * decayExponent * 0.5))
            values[n] = logf(a / (1.0 - a))
            n += 1
        }
        return values
    }

    /// 数値的に安定な log σ(x)
    public static func logSigmoid(_ x: Float) -> Float {
        if x < 0.0 {
            return x - log1pf(expf(x))
        }
        return -log1pf(expf(-x))
    }
}
