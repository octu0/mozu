import Foundation

/// コーパスの音素アライメント（MAS）の作成とキャッシュ
///
/// 音素列は推論と同じ PhonemeSequenceBuilder から作る（文頭・文末の無音を除いた本体。句間のポーズを含む）。
/// 教師 Mel の音素プロトタイプを自己学習しながら単調アライメントを反復する。
public enum AlignmentBuilder {
    public struct Statistics: Sendable {
        public let loaded: Int
        public let built: Int
        public let skipped: Int
    }

    /// キャッシュがあれば読み、足りない発話だけ作って保存する
    public static func loadOrBuild(
        corpus: Corpus,
        sequenceBuilder: PhonemeSequenceBuilder,
        force: Bool = false,
        iterations: Int = 3,
        log: (String) -> Void = { print($0) }
    ) throws -> (alignments: [String: UtteranceAlignment], statistics: Statistics) {
        var cached: [String: UtteranceAlignment] = [:]
        if force != true, FileManager.default.fileExists(atPath: corpus.alignmentPath) {
            cached = (try? AlignmentStore.load(from: corpus.alignmentPath)) ?? [:]
        }
        // 音素列が変わった（語彙の更新など）発話はキャッシュを使わない
        var pending: [CorpusUtterance] = []
        var alignments: [String: UtteranceAlignment] = [:]
        for u in corpus.utterances {
            let body = bodyTokens(sequenceBuilder.build(text: u.text))
            if let a = cached[u.id], a.phonemes.count == body.count, matches(a, body) {
                alignments[u.id] = a
            } else {
                pending.append(u)
            }
        }
        let loaded = alignments.count
        var skipped = 0
        if pending.isEmpty != true {
            if pending.count <= modelSubsetSize {
                log("アライメント生成: \(pending.count) 発話（キャッシュ済み \(loaded) 発話）の特徴量を抽出しています...")
                let items = buildItems(utterances: pending, sequenceBuilder: sequenceBuilder)
                skipped = pending.count - items.count
                log("MAS を開始します: \(items.count) 発話（読み込み失敗・不整合で除外 \(skipped) 発話）")
                let results = MonotonicAlignmentSearch.iterativelyAlign(items: items, iterations: iterations, meanFramesPerMora: 16.0)
                for a in results {
                    alignments[a.utteranceId] = a
                }
            } else {
                // 大規模コーパス: 全発話の Mel を同時に持てないので、部分集合でプロトタイプを学習し、残りを分割して整列する
                let stride = max(1, pending.count / modelSubsetSize)
                let subset = pending.enumerated().filter { $0.offset % stride == 0 }.map { $0.element }
                log("アライメント生成: \(pending.count) 発話。まず \(subset.count) 発話で音素プロトタイプを学習します...")
                let subsetItems = buildItems(utterances: subset, sequenceBuilder: sequenceBuilder)
                let aligner = MonotonicAlignmentSearch.trainAligner(items: subsetItems, iterations: iterations, meanFramesPerMora: 16.0)
                var start = 0
                while start < pending.count {
                    let end = min(pending.count, start + chunkSize)
                    let chunk = Array(pending[start..<end])
                    let items = buildItems(utterances: chunk, sequenceBuilder: sequenceBuilder)
                    skipped += chunk.count - items.count
                    let results = aligner.alignUtterances(items: items, meanFramesPerMora: 16.0)
                    for a in results {
                        alignments[a.utteranceId] = a
                    }
                    log("  整列 \(end)/\(pending.count) 発話（除外 \(skipped) 発話）")
                    start = end
                    if end % (chunkSize * 10) == 0 || end == pending.count {
                        try AlignmentStore.save(Array(alignments.values).sorted { $0.utteranceId < $1.utteranceId }, to: corpus.alignmentPath)
                    }
                }
            }
            try AlignmentStore.save(Array(alignments.values).sorted { $0.utteranceId < $1.utteranceId }, to: corpus.alignmentPath)
            log("アライメントを保存しました: \(corpus.alignmentPath) (\(alignments.count) 発話)")
        }
        return (alignments, Statistics(loaded: loaded, built: alignments.count - loaded, skipped: skipped))
    }

    /// プロトタイプ学習に使う発話数の上限（これを超えるコーパスは分割して整列する）
    public static let modelSubsetSize = 4000
    /// 分割整列の 1 回分の発話数
    public static let chunkSize = 2000

    /// 文頭・文末の無音を除いた音素列
    public static func bodyTokens(_ sequence: PhonemeSequence) -> [PhonemeToken] {
        if sequence.tokens.count < 2 {
            return []
        }
        return Array(sequence.tokens[1..<(sequence.tokens.count - 1)])
    }

    static func matches(_ alignment: UtteranceAlignment, _ tokens: [PhonemeToken]) -> Bool {
        var i = 0
        while i < tokens.count {
            if Int(alignment.phonemes[i].phoneId) != tokens[i].id {
                return false
            }
            i += 1
        }
        return true
    }

    /// 発話ごとの Mel・有声フラグ・音素列を並列に用意する
    static func buildItems(utterances: [CorpusUtterance], sequenceBuilder: PhonemeSequenceBuilder) -> [MonotonicAlignmentSearch.AlignmentInputItem] {
        let reader = WavAudioReader()
        let melExtractor = MelSpectrogramExtractor()
        let tracker = PitchTracker()
        var results = [MonotonicAlignmentSearch.AlignmentInputItem?](repeating: nil, count: utterances.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: utterances.count) { index in
            let u = utterances[index]
            let tokens = bodyTokens(sequenceBuilder.build(text: u.text))
            if tokens.isEmpty {
                return
            }
            guard let pcm = Corpus.loadPCM(path: u.wavPath, reader: reader) else {
                return
            }
            let mel = melExtractor.extractLogMel(pcm: pcm)
            let totalFrames = max(1, mel.count)
            let boundaries = SpeechBoundaries.detect(pcm: pcm, totalFrames: totalFrames)
            if boundaries.speechFrames < tokens.count * 2 {
                return
            }
            let pitch = tracker.track(pcm: pcm)
            let speechEnd = min(mel.count, boundaries.leadSilence + boundaries.speechFrames)
            var speechMel: [[Float]] = []
            var speechVoiced: [Float] = []
            var f = boundaries.leadSilence
            while f < speechEnd {
                speechMel.append(mel[f])
                speechVoiced.append(f < pitch.frameCount ? pitch.voiced[f] : 0.0)
                f += 1
            }
            let item = MonotonicAlignmentSearch.AlignmentInputItem(
                utteranceId: u.id,
                leadSilence: boundaries.leadSilence,
                trailSilence: boundaries.trailSilence,
                totalSpeechFrames: speechMel.count,
                mel: speechMel,
                voiced: speechVoiced,
                phonemes: tokens
            )
            lock.lock()
            results[index] = item
            lock.unlock()
        }
        return results.compactMap { $0 }
    }
}
