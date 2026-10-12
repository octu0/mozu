import Foundation

/// ボコーダの設定
///
/// Mel（10 ms フレーム）を双方向 Hawk で処理し、フレームごとに STFT の振幅と位相を出し、逆 STFT で波形にする。
/// パルス列や上採樣畳み込みを使わないので、フレーム周期の櫛（ブザー感）が構造的に出ない。
public struct VocoderConfig: Sendable, Codable, Equatable {
    public let melChannels: Int
    public let modelDim: Int
    public let stack: HawkStackConfig
    /// 合成 STFT の窓長（hop の整数倍）
    public let fftSize: Int
    public let hopSize: Int

    public init(melChannels: Int, modelDim: Int, stack: HawkStackConfig, fftSize: Int, hopSize: Int) {
        precondition(stack.modelDim == modelDim)
        precondition(fftSize % hopSize == 0, "fftSize must be a multiple of hopSize")
        self.melChannels = melChannels
        self.modelDim = modelDim
        self.stack = stack
        self.fftSize = fftSize
        self.hopSize = hopSize
    }

    public var bins: Int {
        return fftSize / 2 + 1
    }

    /// ヘッド出力の次元（振幅 bins + 位相 bins）
    public var headDim: Int {
        return 2 * bins
    }

    /// 1 合成フレームに含まれる hop の数（フレーム位置の周期）
    ///
    /// なぜフレーム位置を入力に与えるか: 周波数 k·sr/fftSize の正弦波は、フレームごとに位相が 2πk·hop/fftSize ずつ進む。
    /// これはフレーム番号 mod (fftSize/hop) で決まる固定パターンなので、位置を与えれば隣接フレームで整合した位相を
    /// 出しやすくなる。位置が無いと「毎フレーム同じ位相」という自明解（10 ms 周期のパルス列＝ブザー）に落ちやすい。
    public var positionPeriod: Int {
        return fftSize / hopSize
    }
}

public struct VocoderWeights: Sendable, Codable, Equatable {
    public var config: VocoderConfig
    public var inputProjection: Tensor   // [melChannels, modelDim]
    public var inputBias: Tensor         // [modelDim]
    public var stack: HawkStackWeights
    public var outputNorm: Tensor        // [modelDim]
    public var headProjection: Tensor    // [modelDim, headDim]
    public var headBias: Tensor          // [headDim]
    /// フレーム位置（frame mod positionPeriod）の埋め込み [positionPeriod, modelDim]
    public var positionEmbedding: Tensor

    public init(config: VocoderConfig, inputProjection: Tensor, inputBias: Tensor, stack: HawkStackWeights, outputNorm: Tensor, headProjection: Tensor, headBias: Tensor, positionEmbedding: Tensor) {
        self.config = config
        self.inputProjection = inputProjection
        self.inputBias = inputBias
        self.stack = stack
        self.outputNorm = outputNorm
        self.headProjection = headProjection
        self.headBias = headBias
        self.positionEmbedding = positionEmbedding
    }

    public var parameterCount: Int {
        return inputProjection.count + inputBias.count + stack.parameterCount + outputNorm.count + headProjection.count + headBias.count + positionEmbedding.count
    }

    enum CodingKeys: String, CodingKey {
        case config, inputProjection, inputBias, stack, outputNorm, headProjection, headBias, positionEmbedding
    }

    /// 位置埋め込みが無い古い重みは 0（位置なし）として読む
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        config = try c.decode(VocoderConfig.self, forKey: .config)
        inputProjection = try c.decode(Tensor.self, forKey: .inputProjection)
        inputBias = try c.decode(Tensor.self, forKey: .inputBias)
        stack = try c.decode(HawkStackWeights.self, forKey: .stack)
        outputNorm = try c.decode(Tensor.self, forKey: .outputNorm)
        headProjection = try c.decode(Tensor.self, forKey: .headProjection)
        headBias = try c.decode(Tensor.self, forKey: .headBias)
        positionEmbedding = try c.decodeIfPresent(Tensor.self, forKey: .positionEmbedding) ?? Tensor(zeros: [config.positionPeriod, config.modelDim])
    }
}

/// 逆 STFT の定数（MLX 学習と CPU 推論で同じ式を使う）
public enum SynthesisSTFT {
    /// 周期 Hann 窓
    public static func window(size: Int) -> [Float] {
        var w = [Float](repeating: 0.0, count: size)
        var n = 0
        while n < size {
            w[n] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(size)))
            n += 1
        }
        return w
    }

    /// 実数逆 DFT の基底。frame[n] = Σ_k re[k]·cosBasis[k][n] + im[k]·sinBasis[k][n]
    /// 行優先 [bins, fftSize]
    public static func inverseBasis(fftSize: Int) -> (cos: [Float], sin: [Float]) {
        let bins = fftSize / 2 + 1
        var c = [Float](repeating: 0.0, count: bins * fftSize)
        var s = [Float](repeating: 0.0, count: bins * fftSize)
        var k = 0
        while k < bins {
            let scale: Float = (k == 0 || k == fftSize / 2) ? 1.0 / Float(fftSize) : 2.0 / Float(fftSize)
            var n = 0
            while n < fftSize {
                let theta = 2.0 * Float.pi * Float(k) * Float(n) / Float(fftSize)
                c[k * fftSize + n] = scale * cos(theta)
                s[k * fftSize + n] = -scale * sin(theta)
                n += 1
            }
            k += 1
        }
        return (c, s)
    }

    /// 合成フレーム t の先頭サンプル。Mel フレーム t（分析窓 frameSize の中心 t·hop + frameSize/2）に中心を合わせる
    public static func frameStart(frame: Int, fftSize: Int, hopSize: Int, analysisFrameSize: Int) -> Int {
        return frame * hopSize + analysisFrameSize / 2 - fftSize / 2
    }

    /// 窓の二乗和（重なり合いの正規化）。出力長 frames·hop に対して返す
    public static func windowSquareSum(frames: Int, fftSize: Int, hopSize: Int, analysisFrameSize: Int) -> [Float] {
        let w = window(size: fftSize)
        let total = frames * hopSize
        var sum = [Float](repeating: 0.0, count: total)
        var t = 0
        while t < frames {
            let start = frameStart(frame: t, fftSize: fftSize, hopSize: hopSize, analysisFrameSize: analysisFrameSize)
            var n = 0
            while n < fftSize {
                let idx = start + n
                if 0 <= idx && idx < total {
                    sum[idx] += w[n] * w[n]
                }
                n += 1
            }
            t += 1
        }
        return sum.map { max($0, 1e-3) }
    }
}
