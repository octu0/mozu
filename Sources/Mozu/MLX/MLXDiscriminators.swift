#if canImport(MLX)
import Foundation
import MLX
import MLXNN

/// 多周期識別器（HiFi-GAN / Vocos の MPD）の 1 周期分
///
/// 波形を周期 p で折り返した 2 次元画像 [B, N/p, p, 1] として畳み込む。周期構造（ピッチ）の自然さを見る
final class MLXPeriodDiscriminator: Module, @unchecked Sendable {
    let period: Int
    var convs: [Conv2d]
    var output: Conv2d

    init(period: Int) {
        self.period = period
        let channels = [1, 32, 128, 256, 512]
        var built: [Conv2d] = []
        var i = 0
        while i + 1 < channels.count {
            built.append(Conv2d(inputChannels: channels[i], outputChannels: channels[i + 1], kernelSize: IntOrPair((5, 1)), stride: IntOrPair((3, 1)), padding: IntOrPair((2, 0))))
            i += 1
        }
        built.append(Conv2d(inputChannels: 512, outputChannels: 512, kernelSize: IntOrPair((5, 1)), stride: IntOrPair((1, 1)), padding: IntOrPair((2, 0))))
        self.convs = built
        self.output = Conv2d(inputChannels: 512, outputChannels: 1, kernelSize: IntOrPair((3, 1)), stride: IntOrPair((1, 1)), padding: IntOrPair((1, 0)))
        super.init()
    }

    /// x: [B, N] → (判定 [B, M], 特徴マップ)
    func callAsFunction(_ x: MLXArray) -> (score: MLXArray, features: [MLXArray]) {
        let b = x.shape[0]
        let n = x.shape[1]
        let pad = (period - n % period) % period
        var w = x
        if 0 < pad {
            w = concatenated([x, MLXArray.zeros([b, pad])], axis: 1)
        }
        var h = w.reshaped([b, (n + pad) / period, period, 1])
        var features: [MLXArray] = []
        for conv in convs {
            h = leakyRelu(conv(h), negativeSlope: 0.1)
            features.append(h)
        }
        h = output(h)
        features.append(h)
        return (h.reshaped([b, -1]), features)
    }
}

/// 多解像度識別器（Vocos の MRD）の 1 解像度分。振幅スペクトログラム [B, T, bins, 1] を畳み込む
final class MLXResolutionDiscriminator: Module, @unchecked Sendable {
    let stft: MLXSTFT
    var convs: [Conv2d]
    var output: Conv2d

    init(fftSize: Int, hopSize: Int) {
        self.stft = MLXSTFT(fftSize: fftSize, hopSize: hopSize)
        let c = 32
        self.convs = [
            Conv2d(inputChannels: 1, outputChannels: c, kernelSize: IntOrPair((3, 9)), stride: IntOrPair((1, 1)), padding: IntOrPair((1, 4))),
            Conv2d(inputChannels: c, outputChannels: c, kernelSize: IntOrPair((3, 9)), stride: IntOrPair((1, 2)), padding: IntOrPair((1, 4))),
            Conv2d(inputChannels: c, outputChannels: c, kernelSize: IntOrPair((3, 9)), stride: IntOrPair((1, 2)), padding: IntOrPair((1, 4))),
            Conv2d(inputChannels: c, outputChannels: c, kernelSize: IntOrPair((3, 9)), stride: IntOrPair((1, 2)), padding: IntOrPair((1, 4))),
            Conv2d(inputChannels: c, outputChannels: c, kernelSize: IntOrPair((3, 3)), stride: IntOrPair((1, 1)), padding: IntOrPair((1, 1))),
        ]
        self.output = Conv2d(inputChannels: c, outputChannels: 1, kernelSize: IntOrPair((3, 3)), stride: IntOrPair((1, 1)), padding: IntOrPair((1, 1)))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (score: MLXArray, features: [MLXArray]) {
        let b = x.shape[0]
        var h = stft.magnitude(x).expandedDimensions(axis: -1)   // [B, T, bins, 1]
        var features: [MLXArray] = []
        for conv in convs {
            h = leakyRelu(conv(h), negativeSlope: 0.1)
            features.append(h)
        }
        h = output(h)
        features.append(h)
        return (h.reshaped([b, -1]), features)
    }
}

/// ボコーダ用の識別器一式（MPD + MRD）
public final class MLXVocoderDiscriminator: Module, @unchecked Sendable {
    var periodDiscriminators: [MLXPeriodDiscriminator]
    var resolutionDiscriminators: [MLXResolutionDiscriminator]

    public init(periods: [Int] = [2, 3, 5, 7, 11], resolutions: [(fft: Int, hop: Int)] = [(256, 64), (512, 128), (1024, 256)]) {
        self.periodDiscriminators = periods.map { MLXPeriodDiscriminator(period: $0) }
        self.resolutionDiscriminators = resolutions.map { MLXResolutionDiscriminator(fftSize: $0.fft, hopSize: $0.hop) }
        super.init()
    }

    /// x: [B, N] → 各識別器の判定と特徴マップ
    public func callAsFunction(_ x: MLXArray) -> [(score: MLXArray, features: [MLXArray])] {
        var out: [(score: MLXArray, features: [MLXArray])] = []
        for d in periodDiscriminators {
            out.append(d(x))
        }
        for d in resolutionDiscriminators {
            out.append(d(x))
        }
        return out
    }

    /// 識別器の損失（ヒンジ）: 本物を +1 側、生成を −1 側へ
    public func discriminatorLoss(real: MLXArray, fake: MLXArray) -> MLXArray {
        let realOut = self(real)
        let fakeOut = self(fake)
        var loss = MLXArray(Float(0))
        var i = 0
        while i < realOut.count {
            loss = loss + mean(relu(1 - realOut[i].score)) + mean(relu(1 + fakeOut[i].score))
            i += 1
        }
        return loss / Float(max(1, realOut.count))
    }

    /// 生成器側の損失: 敵対的（ヒンジ）と特徴一致
    public func generatorLosses(real: MLXArray, fake: MLXArray) -> (adversarial: MLXArray, featureMatching: MLXArray) {
        let realOut = self(stopGradient(real))
        let fakeOut = self(fake)
        var adv = MLXArray(Float(0))
        var fm = MLXArray(Float(0))
        var i = 0
        while i < realOut.count {
            adv = adv + mean(relu(1 - fakeOut[i].score))
            var j = 0
            while j < realOut[i].features.count {
                fm = fm + mean(abs(stopGradient(realOut[i].features[j]) - fakeOut[i].features[j]))
                j += 1
            }
            i += 1
        }
        let n = Float(max(1, realOut.count))
        return (adv / n, fm / n)
    }
}
#endif
