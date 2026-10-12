#if canImport(MLX)
import Foundation
import MLX
import MLXNN

/// 2 層ヘッド（MLX）。重みは CPU 側と同じ [in, out] 行優先
public final class MLXHead: Module, @unchecked Sendable {
    public var w1: MLXArray
    public var b1: MLXArray
    public var w2: MLXArray
    public var b2: MLXArray

    public init(inDim: Int, hidden: Int) {
        let bound1 = sqrt(3.0 / Float(inDim))
        let bound2 = sqrt(3.0 / Float(hidden))
        self.w1 = MLXRandom.uniform(low: -bound1, high: bound1, [inDim, hidden])
        self.b1 = MLXArray.zeros([hidden])
        self.w2 = MLXRandom.uniform(low: -bound2, high: bound2, [hidden, 1])
        self.b2 = MLXArray.zeros([1])
        super.init()
    }

    public func load(_ w: HeadWeights) {
        w1 = MLXArray(w.w1.values, w.w1.shape)
        b1 = MLXArray(w.b1.values, w.b1.shape)
        w2 = MLXArray(w.w2.values, w.w2.shape)
        b2 = MLXArray(w.b2.values, w.b2.shape)
    }

    public func exportWeights() -> HeadWeights {
        return HeadWeights(w1: tensor(w1), b1: tensor(b1), w2: tensor(w2), b2: tensor(b2))
    }

    /// [..., inDim] → [..., 1]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return matmul(hawkGelu(matmul(x, w1) + b1), w2) + b2
    }
}

/// スカラー → 埋め込み（MLX）
public final class MLXScalarProjection: Module, @unchecked Sendable {
    public var w: MLXArray
    public var b: MLXArray

    public init(dim: Int, scale: Float = 0.1) {
        self.w = MLXRandom.uniform(low: -scale, high: scale, [dim])
        self.b = MLXArray.zeros([dim])
        super.init()
    }

    public func load(_ p: ScalarProjection) {
        w = MLXArray(p.w.values, p.w.shape)
        b = MLXArray(p.b.values, p.b.shape)
    }

    public func exportWeights() -> ScalarProjection {
        return ScalarProjection(w: tensor(w), b: tensor(b))
    }

    /// x: [..., 1] → [..., dim]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * w + b
    }
}

/// 音響モデル（MLX、学習用）
public final class MLXAcousticModel: Module, @unchecked Sendable {
    public let config: AcousticConfig
    public var phonemeEmbedding: MLXArray
    public var featureProjection: MLXArray
    public var labelEmbeddings: [MLXArray]
    public var encoder: MLXHawkStack
    public var encoderNorm: MLXArray
    public var durationHead: MLXHead
    public var pitchHead: MLXHead
    public var energyHead: MLXHead
    public var pitchProjection: MLXScalarProjection
    public var energyProjection: MLXScalarProjection
    public var positionProjection: MLXScalarProjection
    public var decoder: MLXHawkStack
    public var decoderNorm: MLXArray
    public var melProjection: MLXArray
    public var melBias: MLXArray

    public init(config: AcousticConfig, labelCounts: [Int]) {
        precondition(labelCounts.count == config.labelKinds.count)
        self.config = config
        let d = config.modelDim
        self.phonemeEmbedding = MLXRandom.normal([config.phonemeVocabSize, d]) * 0.3
        self.featureProjection = MLXRandom.uniform(low: -0.1, high: 0.1, [config.featureCount, d])
        // ラベル埋め込みは 0 から始める（未指定 index 0 と同じ出発点）
        self.labelEmbeddings = labelCounts.map { MLXArray.zeros([$0, d]) }
        self.encoder = MLXHawkStack(config: config.encoder)
        self.encoderNorm = MLXArray.ones([d])
        self.durationHead = MLXHead(inDim: d, hidden: config.headHiddenDim)
        self.pitchHead = MLXHead(inDim: d, hidden: config.headHiddenDim)
        self.energyHead = MLXHead(inDim: d, hidden: config.headHiddenDim)
        self.pitchProjection = MLXScalarProjection(dim: d)
        self.energyProjection = MLXScalarProjection(dim: d)
        self.positionProjection = MLXScalarProjection(dim: d)
        self.decoder = MLXHawkStack(config: config.decoder)
        self.decoderNorm = MLXArray.ones([d])
        let bound = sqrt(3.0 / Float(d))
        self.melProjection = MLXRandom.uniform(low: -bound, high: bound, [d, config.melChannels])
        self.melBias = MLXArray.zeros([config.melChannels])
        super.init()
    }

    public convenience init(weights: AcousticWeights) {
        self.init(config: weights.config, labelCounts: weights.labelEmbeddings.map { $0.shape[0] })
        load(weights)
    }

    public func load(_ w: AcousticWeights) {
        phonemeEmbedding = MLXArray(w.phonemeEmbedding.values, w.phonemeEmbedding.shape)
        featureProjection = MLXArray(w.featureProjection.values, w.featureProjection.shape)
        labelEmbeddings = w.labelEmbeddings.map { MLXArray($0.values, $0.shape) }
        var l = 0
        while l < w.encoder.layers.count {
            encoder.layers[l].load(w.encoder.layers[l])
            l += 1
        }
        encoderNorm = MLXArray(w.encoderNorm.values, w.encoderNorm.shape)
        durationHead.load(w.durationHead)
        pitchHead.load(w.pitchHead)
        energyHead.load(w.energyHead)
        pitchProjection.load(w.pitchProjection)
        energyProjection.load(w.energyProjection)
        positionProjection.load(w.positionProjection)
        l = 0
        while l < w.decoder.layers.count {
            decoder.layers[l].load(w.decoder.layers[l])
            l += 1
        }
        decoderNorm = MLXArray(w.decoderNorm.values, w.decoderNorm.shape)
        melProjection = MLXArray(w.melProjection.values, w.melProjection.shape)
        melBias = MLXArray(w.melBias.values, w.melBias.shape)
    }

    public func exportWeights(config exportConfig: AcousticConfig? = nil) -> AcousticWeights {
        return AcousticWeights(
            config: exportConfig ?? config,
            phonemeEmbedding: tensor(phonemeEmbedding),
            featureProjection: tensor(featureProjection),
            labelEmbeddings: labelEmbeddings.map { tensor($0) },
            encoder: encoder.exportWeights(),
            encoderNorm: tensor(encoderNorm),
            durationHead: durationHead.exportWeights(),
            pitchHead: pitchHead.exportWeights(),
            energyHead: energyHead.exportWeights(),
            pitchProjection: pitchProjection.exportWeights(),
            energyProjection: energyProjection.exportWeights(),
            positionProjection: positionProjection.exportWeights(),
            decoder: decoder.exportWeights(),
            decoderNorm: tensor(decoderNorm),
            melProjection: tensor(melProjection),
            melBias: tensor(melBias)
        )
    }

    /// 学習用バッチ（CPU 側で詰め物をして作る）
    public struct Batch {
        public let phonemeIds: MLXArray      // [B, P] int32
        public let features: MLXArray        // [B, P, F]
        public let labelIds: MLXArray        // [B, K] int32
        public let phoneMask: MLXArray       // [B, P, 1]
        public let voicedMask: MLXArray      // [B, P, 1]
        public let logDurations: MLXArray    // [B, P, 1] 教師 log フレーム数
        public let pitch: MLXArray           // [B, P, 1] 正規化済み
        public let energy: MLXArray          // [B, P, 1] 正規化済み
        public let frameIndex: MLXArray      // [B·T] int32（[B·P] の行への index）
        public let position: MLXArray        // [B, T, 1]
        public let frameMask: MLXArray       // [B, T, 1]
        public let mel: MLXArray             // [B, T, melChannels]
        public let batchSize: Int
        public let phoneCount: Int
        public let frameCount: Int
    }

    public struct Prediction {
        public let mel: MLXArray            // [B, T, melChannels]
        public let logDurations: MLXArray   // [B, P, 1]
        public let pitch: MLXArray          // [B, P, 1]
        public let energy: MLXArray         // [B, P, 1]
    }

    /// 教師強制（継続時間・ピッチ・エネルギーは教師値）で順伝播する
    public func forward(_ batch: Batch) -> Prediction {
        let d = config.modelDim
        var x = take(phonemeEmbedding, batch.phonemeIds, axis: 0)              // [B, P, D]
        x = x + matmul(batch.features, featureProjection)
        var k = 0
        while k < labelEmbeddings.count {
            let ids = batch.labelIds[0..., k]                                    // [B]
            let style = take(labelEmbeddings[k], ids, axis: 0).reshaped([batch.batchSize, 1, d])
            x = x + style
            k += 1
        }
        let encoded = hawkRMSNorm(encoder(x, frameMask: batch.phoneMask), encoderNorm)
        // 継続時間ヘッドはエンコーダへ勾配を流さない（継続時間の誤差で Mel 向けの特徴を歪めないため）
        let logDurations = durationHead(stopGradient(encoded))
        let pitch = pitchHead(encoded)
        let energy = energyHead(encoded)
        let conditioned = encoded + pitchProjection(batch.pitch) + energyProjection(batch.energy)
        let flat = conditioned.reshaped([batch.batchSize * batch.phoneCount, d])
        var frames = take(flat, batch.frameIndex, axis: 0).reshaped([batch.batchSize, batch.frameCount, d])
        frames = frames + positionProjection(batch.position)
        let decoded = hawkRMSNorm(decoder(frames, frameMask: batch.frameMask), decoderNorm)
        let mel = matmul(decoded, melProjection) + melBias
        return Prediction(mel: mel, logDurations: logDurations, pitch: pitch, energy: energy)
    }
}
#endif
