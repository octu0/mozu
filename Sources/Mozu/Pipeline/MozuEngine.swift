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

    /// 声の指定（"female,ギャル,age=10代後半" など）を、重みのラベル語彙で解決する
    ///
    /// "=" の無い値は、その値を持つラベルの種類を探して当てはめる。ラベル語彙の無い重みでは従来のプリセットとして扱う。
    public func voice(named spec: String) throws -> VoiceProfile {
        switch spec.lowercased().trimmingCharacters(in: .whitespaces) {
        case "", "neutral", "default":
            return .neutral
        default:
            break
        }
        if weights.labels.kinds.isEmpty {
            return VoiceProfile.preset(named: spec)
        }
        return VoiceProfile(name: spec, labels: try weights.labels.resolve(spec))
    }

    /// 話し方のクラスタ名から潜在変数を引く（無ければ nil）
    public func styleLatent(named name: String) -> [Float]? {
        return weights.acoustic?.style?.clusters.first { $0.name == name }?.latent
    }

    /// 参照音声の話し方の潜在変数（参照エンコーダの平均）。話し方を持たない重みでは nil
    public func styleLatent(reference pcm: [Float]) -> [Float]? {
        guard let style = weights.acoustic?.style else {
            return nil
        }
        return StyleEncoder(weights: style).encode(mel: MelSpectrogramExtractor().extractLogMel(pcm: pcm)).mean
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
            pitchShift: log2(pitchScale),
            intonation: VoiceProfile.clamp(voice.intonation, 0.5, 2.0, fallback: 1.0),
            energyShift: VoiceProfile.clamp(voice.energy, -2.0, 2.0, fallback: 0.0),
            latent: voice.latent
        )
        return (sequence, output)
    }

    /// 波形まで合成する（16 kHz、-1..1）。モデルが無ければ空
    public func synthesize(text: String, voice: VoiceProfile = .default, speed: Float = 1.0, pitch: Float = 1.0) -> [Float] {
        guard let vocoder = vocoder, let result = synthesizeMel(text: text, voice: voice, speed: speed, pitch: pitch) else {
            return []
        }
        return vocode(mel: result.output.mel, fadeOutFrames: result.output.durations.last ?? 0)
    }

    /// 文末の無音の区間で波形を 0 まで絞る
    ///
    /// 合成した無音は完全な無音ではなく小さな雑音なので、そのまま切れると音声が途中で途切れたように聞こえる。
    static func fadeOut(_ samples: [Float], frames: Int) -> [Float] {
        let length = min(samples.count, frames * AudioConfig.hopSize)
        if length <= 1 {
            return samples
        }
        var out = samples
        let start = samples.count - length
        var i = 0
        while i < length {
            let x = Float(i) / Float(length - 1)
            out[start + i] *= 0.5 * (1.0 + cosf(Float.pi * x))
            i += 1
        }
        return out
    }

    /// Mel から波形を作る。fadeOutFrames: 末尾のこのフレーム数（文末の無音）で波形を絞る
    public func vocode(mel: [[Float]], fadeOutFrames: Int = 0) -> [Float] {
        guard let vocoder = vocoder else {
            return []
        }
        return Self.limit(Self.fadeOut(vocoder.synthesize(mel: mel), frames: fadeOutFrames))
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
