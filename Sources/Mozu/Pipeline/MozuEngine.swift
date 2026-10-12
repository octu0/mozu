import Foundation

/// mozu 音声合成エンジン（CPU 推論）
///
/// テキスト → 形態素解析・アクセント句 → 音素列 → 音響モデル（Mel）→ ボコーダ（波形）。
public final class MozuEngine: @unchecked Sendable {
    public let weights: MozuWeights
    public let normalizer: TextNormalizer
    public let prosodyModel: ProsodyModel
    public let vocabulary: PhonemeVocabulary
    public let sequenceBuilder: PhonemeSequenceBuilder
    public let acoustic: AcousticModel?
    public let vocoder: Vocoder?

    public var sampleRate: Float {
        return Float(AudioConfig.sampleRate)
    }

    public init(weights: MozuWeights) {
        self.weights = weights
        self.vocabulary = PhonemeVocabulary()
        self.normalizer = TextNormalizer(morphology: ViterbiMorphology(lexicon: weights.lexicon.isEmpty ? nil : weights.lexicon))
        self.prosodyModel = ProsodyModel()
        self.sequenceBuilder = PhonemeSequenceBuilder(normalizer: normalizer, prosodyModel: prosodyModel, vocabulary: vocabulary)
        self.acoustic = weights.acoustic.map { AcousticModel(weights: $0) }
        self.vocoder = weights.vocoder.map { Vocoder(weights: $0) }
    }

    /// 重み無し（言語処理のみ）
    public convenience init() {
        self.init(weights: MozuWeights())
    }

    public static func load(from path: String) throws -> MozuEngine {
        return MozuEngine(weights: try MozuWeights.load(from: URL(fileURLWithPath: path)))
    }

    /// Mel まで合成する。音響モデルが無ければ nil
    public func synthesizeMel(text: String, voice: VoiceProfile = .default, speed: Float = 1.0, pitch: Float = 1.0) -> (sequence: PhonemeSequence, output: AcousticModel.Output)? {
        guard let acoustic = acoustic else {
            return nil
        }
        let sequence = sequenceBuilder.build(text: text)
        if sequence.count < 3 {
            return nil
        }
        let labelIndices = weights.labels.indices(for: voice.labels)
        let speedScale = VoiceProfile.clamp(speed, 0.5, 2.0, fallback: 1.0) * voice.speed
        let pitchScale = VoiceProfile.clamp(pitch, 0.5, 2.0, fallback: 1.0) * voice.pitch
        let output = acoustic.synthesize(
            phonemeIds: sequence.ids,
            features: sequence.features,
            labelIndices: labelIndices,
            speed: speedScale,
            pitchShift: log2(pitchScale)
        )
        return (sequence, output)
    }

    /// 波形まで合成する（16 kHz、-1..1）。モデルが無ければ空
    public func synthesize(text: String, voice: VoiceProfile = .default, speed: Float = 1.0, pitch: Float = 1.0) -> [Float] {
        guard let vocoder = vocoder, let result = synthesizeMel(text: text, voice: voice, speed: speed, pitch: pitch) else {
            return []
        }
        return Self.limit(vocoder.synthesize(mel: result.output.mel))
    }

    /// Mel から波形を作る（コピー合成・診断用）
    public func vocode(mel: [[Float]]) -> [Float] {
        guard let vocoder = vocoder else {
            return []
        }
        return Self.limit(vocoder.synthesize(mel: mel))
    }

    public func synthesizeWav(text: String, voice: VoiceProfile = .default, speed: Float = 1.0, pitch: Float = 1.0) -> Data {
        let samples = synthesize(text: text, voice: voice, speed: speed, pitch: pitch)
        return WavEncoder.encode(samples: samples, sampleRate: AudioConfig.sampleRate)
    }

    /// 波形を区切って順に渡す。返り値は総サンプル数
    @discardableResult
    public func synthesizeStream(
        text: String,
        voice: VoiceProfile = .default,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        chunkSamples: Int = 3200,
        isCancelled: () -> Bool = { false },
        onFrame: ([Float]) -> Void
    ) -> Int {
        let samples = synthesize(text: text, voice: voice, speed: speed, pitch: pitch)
        var start = 0
        while start < samples.count {
            if isCancelled() {
                break
            }
            let end = min(samples.count, start + max(1, chunkSamples))
            onFrame(Array(samples[start..<end]))
            start = end
        }
        return samples.count
    }

    /// クリップ防止（ピークが 0.95 を超えるときだけ縮める）
    static func limit(_ samples: [Float]) -> [Float] {
        var peak: Float = 0.0
        for v in samples {
            let a = abs(v)
            if peak < a {
                peak = a
            }
        }
        if peak <= 0.95 || peak.isFinite != true {
            return samples.map { $0.isFinite ? $0 : 0.0 }
        }
        let scale = 0.95 / peak
        return samples.map { $0.isFinite ? $0 * scale : 0.0 }
    }
}
