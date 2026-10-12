import Foundation

/// 音響モデル学習の 1 発話分の教師データ
public struct UtteranceSample: Sendable {
    public let id: String
    public let text: String
    /// 文頭・文末の無音を含む音素 ID 列
    public let phonemeIds: [Int32]
    /// [phoneCount][featureCount]
    public let features: [[Float]]
    /// 条件付けラベルの index（LabelVocabulary.kinds の順）
    public let labelIndices: [Int]
    /// 音素ごとのフレーム数（合計 = frameCount）
    public let durations: [Int]
    /// 音素ごとの log2(F0 / 220)（無声音素は 0）
    public let pitch: [Float]
    /// 音素が有声（F0 教師が存在する）か
    public let voiced: [Bool]
    /// 音素ごとの有声フレーム数（アクセント核の学習で信頼度に使う）
    public let voicedFrames: [Int]
    /// 音素ごとの平均対数 Mel（エネルギー）
    public let energy: [Float]
    /// [frameCount][melChannels]。遅延読み込みのときは空
    public let mel: [[Float]]
    /// 16 kHz PCM（ボコーダ学習用。音響モデル学習では空にできる）
    public let pcm: [Float]
    /// Mel のフレーム数（mel が空でも分かるように持つ）
    public let frames: Int
    /// 遅延読み込みのための音声ファイル
    public let wavPath: String

    public var frameCount: Int {
        return frames
    }

    public var phoneCount: Int {
        return phonemeIds.count
    }

    /// アクセント特徴だけを差し替える（語彙の核が更新されたとき）
    public func withFeatures(_ newFeatures: [[Float]]) -> UtteranceSample {
        return UtteranceSample(id: id, text: text, phonemeIds: phonemeIds, features: newFeatures, labelIndices: labelIndices, durations: durations, pitch: pitch, voiced: voiced, voicedFrames: voicedFrames, energy: energy, mel: mel, pcm: pcm, frames: frames, wavPath: wavPath)
    }

    public init(id: String, text: String, phonemeIds: [Int32], features: [[Float]], labelIndices: [Int], durations: [Int], pitch: [Float], voiced: [Bool], voicedFrames: [Int], energy: [Float], mel: [[Float]], pcm: [Float], frames: Int? = nil, wavPath: String = "") {
        self.id = id
        self.text = text
        self.phonemeIds = phonemeIds
        self.features = features
        self.labelIndices = labelIndices
        self.durations = durations
        self.pitch = pitch
        self.voiced = voiced
        self.voicedFrames = voicedFrames
        self.energy = energy
        self.mel = mel
        self.pcm = pcm
        self.frames = frames ?? mel.count
        self.wavPath = wavPath
    }
}

/// コーパスとアライメントから学習サンプルを作る
public enum SampleBuilder {
    public static let referenceF0: Float = 220.0

    public struct Options: Sendable {
        public var keepPCM: Bool
        public var maxFrames: Int
        /// Mel をメモリに持つ。false のときは学習時にバッチごとに読み直す（大規模コーパス向け）
        public var keepMel: Bool

        public init(keepPCM: Bool = false, maxFrames: Int = 2000, keepMel: Bool = true) {
            self.keepPCM = keepPCM
            self.maxFrames = maxFrames
            self.keepMel = keepMel
        }
    }

    /// 遅延読み込み: サンプルの音声を読み直して Mel を作る（並列）
    public static func loadMels(samples: [UtteranceSample]) -> [[[Float]]] {
        let reader = WavAudioReader()
        let extractor = MelSpectrogramExtractor()
        var out = [[[Float]]](repeating: [], count: samples.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: samples.count) { i in
            let s = samples[i]
            if s.mel.isEmpty != true {
                lock.lock(); out[i] = s.mel; lock.unlock()
                return
            }
            guard let pcm = Corpus.loadPCM(path: s.wavPath, reader: reader) else {
                return
            }
            var mel = extractor.extractLogMel(pcm: pcm)
            if mel.count != s.frames {
                // 再読込で長さが違うことは無いはずだが、念のため合わせる
                if mel.count > s.frames {
                    mel = Array(mel.prefix(s.frames))
                } else {
                    let pad = [Float](repeating: log(MelSpectrogramExtractor.logMelFloor), count: AudioConfig.melChannels)
                    while mel.count < s.frames { mel.append(pad) }
                }
            }
            lock.lock(); out[i] = mel; lock.unlock()
        }
        return out
    }

    public static func build(
        corpus: Corpus,
        alignments: [String: UtteranceAlignment],
        labels: LabelVocabulary,
        sequenceBuilder: PhonemeSequenceBuilder,
        options: Options = Options()
    ) -> [UtteranceSample] {
        let reader = WavAudioReader()
        let melExtractor = MelSpectrogramExtractor()
        let tracker = PitchTracker()
        var results = [UtteranceSample?](repeating: nil, count: corpus.utterances.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: corpus.utterances.count) { index in
            let u = corpus.utterances[index]
            guard let alignment = alignments[u.id] else {
                return
            }
            let sequence = sequenceBuilder.build(text: u.text)
            let body = AlignmentBuilder.bodyTokens(sequence)
            if body.count != alignment.phonemes.count || AlignmentBuilder.matches(alignment, body) != true {
                return
            }
            guard let pcm = Corpus.loadPCM(path: u.wavPath, reader: reader) else {
                return
            }
            let mel = melExtractor.extractLogMel(pcm: pcm)
            let totalFrames = mel.count
            if totalFrames < body.count * 2 || options.maxFrames < totalFrames {
                return
            }
            // 継続時間: [先頭無音] + 本体 + [末尾無音]。合計を Mel のフレーム数に合わせる
            var durations = [max(1, alignment.leadSilenceFrames)]
            durations.append(contentsOf: alignment.phonemes.map { max(1, $0.durationFrames) })
            durations.append(max(1, alignment.trailSilenceFrames))
            let sum = durations.reduce(0, +)
            let diff = totalFrames - sum
            if diff != 0 {
                let last = durations.count - 1
                if 1 <= durations[last] + diff {
                    durations[last] += diff
                } else {
                    durations[last] = 1
                    let rest = totalFrames - (sum - durations[durations.count - 1]) - 1
                    durations[0] = max(1, durations[0] + rest - durations[0])
                    let fixed = durations.reduce(0, +)
                    if fixed != totalFrames {
                        durations[last - 1] = max(1, durations[last - 1] + (totalFrames - fixed))
                    }
                }
            }
            if durations.reduce(0, +) != totalFrames {
                return
            }
            // 音素ごとのピッチ（有声フレームの中央値）とエネルギー
            let pitchTrack = tracker.track(pcm: pcm)
            var pitch: [Float] = []
            var voiced: [Bool] = []
            var voicedFrames: [Int] = []
            var energy: [Float] = []
            var cursor = 0
            var p = 0
            while p < durations.count {
                let dur = durations[p]
                var values: [Float] = []
                var energySum: Float = 0.0
                var f = cursor
                while f < cursor + dur && f < totalFrames {
                    if f < pitchTrack.frameCount && 0.5 <= pitchTrack.voiced[f] && 50.0 < pitchTrack.f0[f] {
                        values.append(log2(pitchTrack.f0[f] / referenceF0))
                    }
                    var m: Float = 0.0
                    for v in mel[f] {
                        m += v
                    }
                    energySum += m / Float(max(1, mel[f].count))
                    f += 1
                }
                values.sort()
                let isVoiced = 2 <= values.count && sequence.tokens[p].category != .pause
                pitch.append(isVoiced ? values[values.count / 2] : 0.0)
                voiced.append(isVoiced)
                voicedFrames.append(values.count)
                energy.append(energySum / Float(max(1, dur)))
                cursor += dur
                p += 1
            }
            let sample = UtteranceSample(
                id: u.id,
                text: u.text,
                phonemeIds: sequence.tokens.map { Int32($0.id) },
                features: sequence.features,
                labelIndices: labels.indices(for: u.labels),
                durations: durations,
                pitch: pitch,
                voiced: voiced,
                voicedFrames: voicedFrames,
                energy: energy,
                mel: options.keepMel ? mel : [],
                pcm: options.keepPCM ? pcm : [],
                frames: totalFrames,
                wavPath: u.wavPath
            )
            lock.lock()
            results[index] = sample
            lock.unlock()
        }
        return results.compactMap { $0 }
    }

    /// ピッチ・エネルギーの正規化定数（有声音素のみ / 全音素）
    public static func normalization(samples: [UtteranceSample]) -> (pitchMean: Float, pitchStd: Float, energyMean: Float, energyStd: Float) {
        var pSum: Double = 0, pSq: Double = 0, pN: Double = 0
        var eSum: Double = 0, eSq: Double = 0, eN: Double = 0
        for s in samples {
            var p = 0
            while p < s.phoneCount {
                if s.voiced[p] {
                    pSum += Double(s.pitch[p]); pSq += Double(s.pitch[p] * s.pitch[p]); pN += 1
                }
                eSum += Double(s.energy[p]); eSq += Double(s.energy[p] * s.energy[p]); eN += 1
                p += 1
            }
        }
        let pMean = pN > 0 ? pSum / pN : 0
        let pStd = pN > 1 ? sqrt(max(1e-6, pSq / pN - pMean * pMean)) : 1
        let eMean = eN > 0 ? eSum / eN : 0
        let eStd = eN > 1 ? sqrt(max(1e-6, eSq / eN - eMean * eMean)) : 1
        return (Float(pMean), Float(pStd), Float(eMean), Float(eStd))
    }
}
