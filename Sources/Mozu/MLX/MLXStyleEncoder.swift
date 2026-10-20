#if canImport(MLX)
import Foundation
import MLX
import MLXNN

/// 参照エンコーダ（MLX、学習用）。実測 Mel → 双方向 Hawk → フレーム平均 → z の平均・対数分散
public final class MLXStyleEncoder: Module, @unchecked Sendable {
    public let latentDim: Int
    public var inputProjection: MLXArray
    public var inputBias: MLXArray
    public var stack: MLXHawkStack
    public var outputNorm: MLXArray
    public var meanProjection: MLXArray
    public var meanBias: MLXArray
    public var logVarProjection: MLXArray
    public var logVarBias: MLXArray
    public var styleProjection: MLXArray
    /// クラスタは学習しないので Module の外の値として持つ
    var clusters: [StyleCluster] = []

    public init(melChannels: Int, modelDim: Int, latentDim: Int = StyleDefaults.latentDim, refDim: Int = StyleDefaults.refDim, layers: Int = StyleDefaults.layers) {
        self.latentDim = latentDim
        let stackConfig = HawkStackConfig(modelDim: refDim, rnnDim: refDim, mlpDim: refDim * 3, heads: 4, convLeft: 2, convRight: 2, numLayers: layers, bidirectional: true)
        let inBound = sqrt(3.0 / Float(melChannels))
        let refBound = sqrt(3.0 / Float(refDim))
        self.inputProjection = MLXRandom.uniform(low: -inBound, high: inBound, [melChannels, refDim])
        self.inputBias = MLXArray.zeros([refDim])
        self.stack = MLXHawkStack(config: stackConfig)
        self.outputNorm = MLXArray.ones([refDim])
        self.meanProjection = MLXRandom.uniform(low: -refBound, high: refBound, [refDim, latentDim])
        self.meanBias = MLXArray.zeros([latentDim])
        // 対数分散は小さく始める（初めは z をほぼ平均どおりに使わせる）
        self.logVarProjection = MLXArray.zeros([refDim, latentDim])
        self.logVarBias = MLXArray.zeros([latentDim]) - 2.0
        // z の影響はラベル埋め込みと同程度の大きさから始める（小さすぎると z を使う方向へ学習が進まない）
        self.styleProjection = MLXRandom.uniform(low: -0.1, high: 0.1, [latentDim, modelDim])
        super.init()
    }

    public convenience init(weights w: StyleWeights) {
        self.init(melChannels: w.inputProjection.shape[0], modelDim: w.styleProjection.shape[1], latentDim: w.latentDim, refDim: w.refDim, layers: w.stack.config.numLayers)
        inputProjection = MLXArray(w.inputProjection.values, w.inputProjection.shape)
        inputBias = MLXArray(w.inputBias.values, w.inputBias.shape)
        stack = MLXHawkStack(weights: w.stack)
        outputNorm = MLXArray(w.outputNorm.values, w.outputNorm.shape)
        meanProjection = MLXArray(w.meanProjection.values, w.meanProjection.shape)
        meanBias = MLXArray(w.meanBias.values, w.meanBias.shape)
        logVarProjection = MLXArray(w.logVarProjection.values, w.logVarProjection.shape)
        logVarBias = MLXArray(w.logVarBias.values, w.logVarBias.shape)
        styleProjection = MLXArray(w.styleProjection.values, w.styleProjection.shape)
        clusters = w.clusters
    }

    public func exportWeights() -> StyleWeights {
        return StyleWeights(latentDim: latentDim, inputProjection: tensor(inputProjection), inputBias: tensor(inputBias), stack: stack.exportWeights(), outputNorm: tensor(outputNorm), meanProjection: tensor(meanProjection), meanBias: tensor(meanBias), logVarProjection: tensor(logVarProjection), logVarBias: tensor(logVarBias), styleProjection: tensor(styleProjection), clusters: clusters)
    }

    /// mel [B, T, melChannels]、frameMask [B, T, 1] → (平均, 対数分散) [B, latentDim]
    public func encode(_ mel: MLXArray, frameMask: MLXArray) -> (mean: MLXArray, logVar: MLXArray) {
        let x = matmul(mel, inputProjection) + inputBias
        let h = hawkRMSNorm(stack(x, frameMask: frameMask), outputNorm)
        let pooled = sum(h * frameMask, axis: 1) / maximum(sum(frameMask, axis: 1), MLXArray(Float(1)))
        let mean = matmul(pooled, meanProjection) + meanBias
        // 対数分散は極端な値で発散しないよう範囲を絞る
        let logVar = clip(matmul(pooled, logVarProjection) + logVarBias, min: -8.0, max: 4.0)
        return (mean, logVar)
    }
}

/// 参照エンコーダの既定構成
public enum StyleDefaults {
    public static let latentDim = 16
    public static let refDim = 128
    public static let layers = 2
}
#endif
