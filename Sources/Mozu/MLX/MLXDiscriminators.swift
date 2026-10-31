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

#if canImport(MLX)
/// 抑揚の識別器（音響モデル用）
///
/// 音素ごとの（ピッチ, エネルギー）の並びが、実音声の値か音響モデルの予測かを見分ける。
/// 条件として音素のエンコーダ出力（勾配は止める）を受け取り、「この文脈でこの抑揚は自然か」を判定する。
/// なぜ必要か: 二乗誤差だけで学習したピッチは平均に寄り、抑揚が実音声の 6 割ほどに縮む。識別器が平板な抑揚を見分けるので、
/// 予測が実音声らしい幅を持つようになる。
public final class MLXProsodyDiscriminator: Module, @unchecked Sendable {
    var condition: Linear
    var convs: [Conv1d]
    var output: Conv1d

    public init(modelDim: Int, channels: Int = 128) {
        self.condition = Linear(modelDim, 64)
        self.convs = [
            Conv1d(inputChannels: 64 + 2, outputChannels: channels, kernelSize: 5, padding: 2),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 5, padding: 4, dilation: 2),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 5, padding: 8, dilation: 4),
        ]
        self.output = Conv1d(inputChannels: channels, outputChannels: 1, kernelSize: 3, padding: 1)
        super.init()
    }

    /// encoded [B, P, D]、prosody [B, P, 2]、phoneMask [B, P, 1] → (音素ごとの判定 [B, P, 1], 特徴マップ)
    public func callAsFunction(encoded: MLXArray, prosody: MLXArray, phoneMask: MLXArray) -> (score: MLXArray, features: [MLXArray]) {
        var h = concatenated([condition(encoded), prosody], axis: -1) * phoneMask
        var features: [MLXArray] = []
        for conv in convs {
            h = leakyRelu(conv(h), negativeSlope: 0.1) * phoneMask
            features.append(h)
        }
        return (output(h) * phoneMask, features)
    }

    /// 識別器の損失（ヒンジ、詰め物の音素は除く）
    public func discriminatorLoss(encoded: MLXArray, real: MLXArray, fake: MLXArray, phoneMask: MLXArray) -> MLXArray {
        let count = maximum(sum(phoneMask), MLXArray(Float(1)))
        let r = self(encoded: encoded, prosody: real, phoneMask: phoneMask).score
        let f = self(encoded: encoded, prosody: fake, phoneMask: phoneMask).score
        return (sum(relu(1 - r) * phoneMask) + sum(relu(1 + f) * phoneMask)) / count
    }

    /// 生成器側の損失: 敵対的（ヒンジ）と特徴一致
    public func generatorLosses(encoded: MLXArray, real: MLXArray, fake: MLXArray, phoneMask: MLXArray) -> (adversarial: MLXArray, featureMatching: MLXArray) {
        let count = maximum(sum(phoneMask), MLXArray(Float(1)))
        let realOut = self(encoded: encoded, prosody: stopGradient(real), phoneMask: phoneMask)
        let fakeOut = self(encoded: encoded, prosody: fake, phoneMask: phoneMask)
        let adv = sum(relu(1 - fakeOut.score) * phoneMask) / count
        var fm = MLXArray(Float(0))
        for (r, f) in zip(realOut.features, fakeOut.features) {
            fm = fm + mean(abs(stopGradient(r) - f))
        }
        return (adv, fm / Float(max(1, realOut.features.count)))
    }
}
#endif

#if canImport(MLX)
/// Mel の識別器（音響モデル用）
///
/// 対数 Mel を時間 × 帯域の画像 [B, T, melChannels, 1] として畳み込み、時間方向はそのまま、帯域方向だけ間引く。
/// なぜ必要か: L1 だけで学習した Mel は時間方向になまり（隣のフレームとの差が実音声の半分ほど）、
/// ボコーダを通すとカサカサ・ロボット的に聞こえる。なまった Mel を識別器が見分けるので、生成側が実音声らしい細かな変化を持つ。
public final class MLXMelDiscriminator: Module, @unchecked Sendable {
    var convs: [Conv2d]
    var output: Conv2d

    public init(channels: Int = 32) {
        let c = channels
        self.convs = [
            Conv2d(inputChannels: 1, outputChannels: c, kernelSize: IntOrPair((5, 5)), stride: IntOrPair((1, 1)), padding: IntOrPair((2, 2))),
            Conv2d(inputChannels: c, outputChannels: c * 2, kernelSize: IntOrPair((5, 5)), stride: IntOrPair((1, 2)), padding: IntOrPair((2, 2))),
            Conv2d(inputChannels: c * 2, outputChannels: c * 2, kernelSize: IntOrPair((5, 5)), stride: IntOrPair((1, 2)), padding: IntOrPair((4, 2)), dilation: IntOrPair((2, 1))),
            Conv2d(inputChannels: c * 2, outputChannels: c * 2, kernelSize: IntOrPair((5, 5)), stride: IntOrPair((1, 2)), padding: IntOrPair((8, 2)), dilation: IntOrPair((4, 1))),
        ]
        self.output = Conv2d(inputChannels: c * 2, outputChannels: 1, kernelSize: IntOrPair((3, 3)), stride: IntOrPair((1, 1)), padding: IntOrPair((1, 1)))
        super.init()
    }

    /// mel [B, T, C]、frameMask [B, T, 1] → (判定 [B, T, C', 1], 特徴マップ)
    public func callAsFunction(_ mel: MLXArray, frameMask: MLXArray) -> (score: MLXArray, features: [MLXArray]) {
        let mask = frameMask.expandedDimensions(axis: -1)    // [B, T, 1, 1]
        var h = (mel * frameMask).expandedDimensions(axis: -1)
        var features: [MLXArray] = []
        for conv in convs {
            h = leakyRelu(conv(h), negativeSlope: 0.1) * mask
            features.append(h)
        }
        return (output(h) * mask, features)
    }

    static func maskedMean(_ x: MLXArray, _ frameMask: MLXArray) -> MLXArray {
        let width = Float(x.shape[2] * x.shape[3])
        return sum(x) / (maximum(sum(frameMask), MLXArray(Float(1))) * width)
    }

    /// 識別器の損失（ヒンジ、詰め物のフレームは除く）
    public func discriminatorLoss(real: MLXArray, fake: MLXArray, frameMask: MLXArray) -> MLXArray {
        let mask = frameMask.expandedDimensions(axis: -1)
        let r = self(real, frameMask: frameMask).score
        let f = self(fake, frameMask: frameMask).score
        return Self.maskedMean(relu(1 - r) * mask, frameMask) + Self.maskedMean(relu(1 + f) * mask, frameMask)
    }

    /// 生成器側の損失: 敵対的（ヒンジ）と特徴一致
    public func generatorLosses(real: MLXArray, fake: MLXArray, frameMask: MLXArray) -> (adversarial: MLXArray, featureMatching: MLXArray) {
        let mask = frameMask.expandedDimensions(axis: -1)
        let realOut = self(stopGradient(real), frameMask: frameMask)
        let fakeOut = self(fake, frameMask: frameMask)
        let adv = Self.maskedMean(relu(1 - fakeOut.score) * mask, frameMask)
        var fm = MLXArray(Float(0))
        for (r, f) in zip(realOut.features, fakeOut.features) {
            fm = fm + Self.maskedMean(abs(stopGradient(r) - f), frameMask)
        }
        return (adv, fm / Float(max(1, realOut.features.count)))
    }
}
#endif
