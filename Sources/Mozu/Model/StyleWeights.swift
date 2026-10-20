import Foundation

/// 話し方の潜在変数（VAE の参照エンコーダ）の重み
///
/// 学習時は、その発話の実測 Mel から潜在変数 z（平均・対数分散）を推定し、z を音響モデルの入力に足す。
/// ラベルでは表せない話し方の違い（声の張り、抑揚の幅、話す速さ、息っぽさなど）が z に入る。
/// 推論時は z = 0（平均的な話し方）、参照音声から推定した平均、またはクラスタの代表点を使う。
public struct StyleWeights: Sendable, Codable, Equatable {
    public var latentDim: Int
    public var inputProjection: Tensor     // [melChannels, refDim]
    public var inputBias: Tensor           // [refDim]
    public var stack: HawkStackWeights
    public var outputNorm: Tensor          // [refDim]
    public var meanProjection: Tensor      // [refDim, latentDim]
    public var meanBias: Tensor            // [latentDim]
    public var logVarProjection: Tensor    // [refDim, latentDim]
    public var logVarBias: Tensor          // [latentDim]
    /// z → 音響モデルの入力 [latentDim, modelDim]
    public var styleProjection: Tensor
    /// 学習データの z を分類した代表点（train styles が作る）
    public var clusters: [StyleCluster]

    public init(latentDim: Int, inputProjection: Tensor, inputBias: Tensor, stack: HawkStackWeights, outputNorm: Tensor, meanProjection: Tensor, meanBias: Tensor, logVarProjection: Tensor, logVarBias: Tensor, styleProjection: Tensor, clusters: [StyleCluster] = []) {
        self.latentDim = latentDim
        self.inputProjection = inputProjection
        self.inputBias = inputBias
        self.stack = stack
        self.outputNorm = outputNorm
        self.meanProjection = meanProjection
        self.meanBias = meanBias
        self.logVarProjection = logVarProjection
        self.logVarBias = logVarBias
        self.styleProjection = styleProjection
        self.clusters = clusters
    }

    public var refDim: Int {
        return stack.config.modelDim
    }

    public var parameterCount: Int {
        return inputProjection.count + inputBias.count + stack.parameterCount + outputNorm.count + meanProjection.count + meanBias.count + logVarProjection.count + logVarBias.count + styleProjection.count
    }
}

/// 話し方のクラスタ（学習データの z の代表点）
public struct StyleCluster: Sendable, Codable, Equatable {
    /// 指定に使う名前（"style3" など）
    public var name: String
    /// 代表点の z
    public var latent: [Float]
    /// 属する学習発話の数
    public var count: Int
    /// このクラスタに特に多いラベル（kind=value）
    public var characteristic: [String]

    public init(name: String, latent: [Float], count: Int, characteristic: [String]) {
        self.name = name
        self.latent = latent
        self.count = count
        self.characteristic = characteristic
    }
}

/// 参照エンコーダの CPU 推論（参照音声から z の平均・対数分散を求める）
public final class StyleEncoder: @unchecked Sendable {
    public let weights: StyleWeights
    private let stack: HawkStack

    public init(weights: StyleWeights) {
        self.weights = weights
        self.stack = HawkStack(weights: weights.stack)
    }

    /// mel: [frames][melChannels] → (平均, 対数分散)
    public func encode(mel: [[Float]]) -> (mean: [Float], logVar: [Float]) {
        let w = weights
        let frames = mel.count
        let r = w.refDim
        let z = w.latentDim
        if frames == 0 {
            return ([Float](repeating: 0, count: z), [Float](repeating: 0, count: z))
        }
        let melDim = w.inputProjection.shape[0]
        var melFlat = [Float](repeating: 0.0, count: frames * melDim)
        var t = 0
        while t < frames {
            var c = 0
            while c < melDim && c < mel[t].count {
                melFlat[t * melDim + c] = mel[t][c]
                c += 1
            }
            t += 1
        }
        var x = [Float](repeating: 0.0, count: frames * r)
        t = 0
        while t < frames {
            var c = 0
            while c < r {
                x[t * r + c] = w.inputBias.values[c]
                c += 1
            }
            t += 1
        }
        Gemm.multiply(melFlat, w.inputProjection.values, &x, m: frames, n: r, k: melDim, beta: 1.0)
        let h = stack.forward(x, frames: frames)
        var normed = [Float](repeating: 0.0, count: frames * r)
        HawkStack.rmsNorm(h, w.outputNorm.values, &normed, frames: frames, width: r)
        var pooled = [Float](repeating: 0.0, count: r)
        t = 0
        while t < frames {
            var c = 0
            while c < r {
                pooled[c] += normed[t * r + c]
                c += 1
            }
            t += 1
        }
        var c = 0
        while c < r {
            pooled[c] /= Float(frames)
            c += 1
        }
        var mean = w.meanBias.values
        var logVar = w.logVarBias.values
        Gemm.multiply(pooled, w.meanProjection.values, &mean, m: 1, n: z, k: r, beta: 1.0)
        Gemm.multiply(pooled, w.logVarProjection.values, &logVar, m: 1, n: z, k: r, beta: 1.0)
        return (mean, logVar)
    }

    /// z → 音響モデルの入力に足すベクトル [modelDim]
    public static func styleVector(_ weights: StyleWeights, latent: [Float], modelDim: Int) -> [Float] {
        var out = [Float](repeating: 0.0, count: modelDim)
        var k = 0
        while k < weights.latentDim && k < latent.count {
            let v = latent[k]
            if v != 0.0 {
                var c = 0
                while c < modelDim {
                    out[c] += v * weights.styleProjection.values[k * modelDim + c]
                    c += 1
                }
            }
            k += 1
        }
        return out
    }
}
