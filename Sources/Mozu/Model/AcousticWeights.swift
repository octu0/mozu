import Foundation

/// 2 層の小さなヘッド（Linear → GeLU → Linear → スカラー）
public struct HeadWeights: Sendable, Codable, Equatable {
    public var w1: Tensor   // [inDim, hidden]
    public var b1: Tensor   // [hidden]
    public var w2: Tensor   // [hidden, 1]
    public var b2: Tensor   // [1]

    public init(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) {
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
    }

    public var parameterCount: Int {
        return w1.count + b1.count + w2.count + b2.count
    }
}

/// スカラーを埋め込みに写す線形層（Linear 1 → dim）
public struct ScalarProjection: Sendable, Codable, Equatable {
    public var w: Tensor   // [dim]
    public var b: Tensor   // [dim]

    public init(w: Tensor, b: Tensor) {
        self.w = w
        self.b = b
    }
}

/// 音響モデルの設定
///
/// テキスト（音素列＋アクセント特徴＋話者ラベル）から Mel を出す非自己回帰モデル。
/// 音素エンコーダ（双方向 Hawk）→ 継続時間・ピッチ・エネルギーの予測 → 長さ調整 → フレームデコーダ（双方向 Hawk）→ Mel。
public struct AcousticConfig: Sendable, Codable, Equatable {
    public let phonemeVocabSize: Int
    /// 音素ごとのアクセント・位置特徴の次元（PhonemeSequenceBuilder.featureCount）
    public let featureCount: Int
    public let modelDim: Int
    public let headHiddenDim: Int
    public let melChannels: Int
    /// 条件付けに使うラベルの種類（順序が埋め込みの順序）
    public let labelKinds: [String]
    public let encoder: HawkStackConfig
    public let decoder: HawkStackConfig
    /// ピッチ目標（log2(F0/220)）の正規化定数
    public let pitchMean: Float
    public let pitchStd: Float
    /// エネルギー目標（音素内の平均対数 Mel）の正規化定数
    public let energyMean: Float
    public let energyStd: Float

    public init(phonemeVocabSize: Int, featureCount: Int, modelDim: Int, headHiddenDim: Int, melChannels: Int, labelKinds: [String], encoder: HawkStackConfig, decoder: HawkStackConfig, pitchMean: Float, pitchStd: Float, energyMean: Float, energyStd: Float) {
        precondition(encoder.modelDim == modelDim && decoder.modelDim == modelDim)
        self.phonemeVocabSize = phonemeVocabSize
        self.featureCount = featureCount
        self.modelDim = modelDim
        self.headHiddenDim = headHiddenDim
        self.melChannels = melChannels
        self.labelKinds = labelKinds
        self.encoder = encoder
        self.decoder = decoder
        self.pitchMean = pitchMean
        self.pitchStd = pitchStd
        self.energyMean = energyMean
        self.energyStd = energyStd
    }

    public func withLabelKinds(_ kinds: [String]) -> AcousticConfig {
        return AcousticConfig(phonemeVocabSize: phonemeVocabSize, featureCount: featureCount, modelDim: modelDim, headHiddenDim: headHiddenDim, melChannels: melChannels, labelKinds: kinds, encoder: encoder, decoder: decoder, pitchMean: pitchMean, pitchStd: pitchStd, energyMean: energyMean, energyStd: energyStd)
    }

    public func withNormalization(pitchMean: Float, pitchStd: Float, energyMean: Float, energyStd: Float) -> AcousticConfig {
        return AcousticConfig(phonemeVocabSize: phonemeVocabSize, featureCount: featureCount, modelDim: modelDim, headHiddenDim: headHiddenDim, melChannels: melChannels, labelKinds: labelKinds, encoder: encoder, decoder: decoder, pitchMean: pitchMean, pitchStd: pitchStd, energyMean: energyMean, energyStd: energyStd)
    }
}

/// 音響モデルの重み
public struct AcousticWeights: Sendable, Codable, Equatable {
    public var config: AcousticConfig
    public var phonemeEmbedding: Tensor        // [phonemeVocabSize, modelDim]
    public var featureProjection: Tensor       // [featureCount, modelDim]
    /// ラベル種類ごとの埋め込み [count + 1, modelDim]（index 0 は未指定）
    public var labelEmbeddings: [Tensor]
    public var encoder: HawkStackWeights
    public var encoderNorm: Tensor             // [modelDim]
    public var durationHead: HeadWeights       // 対数フレーム数
    public var pitchHead: HeadWeights          // 正規化 log2(F0/220)
    public var energyHead: HeadWeights         // 正規化エネルギー
    public var pitchProjection: ScalarProjection
    public var energyProjection: ScalarProjection
    public var positionProjection: ScalarProjection   // 音素内位置 (0..1)
    public var decoder: HawkStackWeights
    public var decoderNorm: Tensor             // [modelDim]
    public var melProjection: Tensor           // [modelDim, melChannels]
    public var melBias: Tensor                 // [melChannels]

    public init(config: AcousticConfig, phonemeEmbedding: Tensor, featureProjection: Tensor, labelEmbeddings: [Tensor], encoder: HawkStackWeights, encoderNorm: Tensor, durationHead: HeadWeights, pitchHead: HeadWeights, energyHead: HeadWeights, pitchProjection: ScalarProjection, energyProjection: ScalarProjection, positionProjection: ScalarProjection, decoder: HawkStackWeights, decoderNorm: Tensor, melProjection: Tensor, melBias: Tensor) {
        self.config = config
        self.phonemeEmbedding = phonemeEmbedding
        self.featureProjection = featureProjection
        self.labelEmbeddings = labelEmbeddings
        self.encoder = encoder
        self.encoderNorm = encoderNorm
        self.durationHead = durationHead
        self.pitchHead = pitchHead
        self.energyHead = energyHead
        self.pitchProjection = pitchProjection
        self.energyProjection = energyProjection
        self.positionProjection = positionProjection
        self.decoder = decoder
        self.decoderNorm = decoderNorm
        self.melProjection = melProjection
        self.melBias = melBias
    }

    public var parameterCount: Int {
        var n = phonemeEmbedding.count + featureProjection.count + encoderNorm.count + decoderNorm.count + melProjection.count + melBias.count
        n += labelEmbeddings.reduce(0) { $0 + $1.count }
        n += encoder.parameterCount + decoder.parameterCount
        n += durationHead.parameterCount + pitchHead.parameterCount + energyHead.parameterCount
        n += pitchProjection.w.count * 2 + energyProjection.w.count * 2 + positionProjection.w.count * 2
        return n
    }
}
