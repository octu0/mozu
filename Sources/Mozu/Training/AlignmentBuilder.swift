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
            if let a = cached[u.id], matches(a, sequenceBuilder.build(text: u.text)) {
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

    /// キャッシュから、今の音素列と合う発話のアライメントだけを読む（作らない・保存しない）
    ///
    /// 発話の一部だけを持つコーパスでも使える（loadOrBuild に一部だけを渡すと、キャッシュがその一部で上書きされる）
    public static func loadCached(corpus: Corpus, sequenceBuilder: PhonemeSequenceBuilder) throws -> [String: UtteranceAlignment] {
        let cached = try AlignmentStore.load(from: corpus.alignmentPath)
        var alignments: [String: UtteranceAlignment] = [:]
        for u in corpus.utterances {
            if let a = cached[u.id], matches(a, sequenceBuilder.build(text: u.text)) {
                alignments[u.id] = a
            }
        }
        return alignments
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

    static func matches(_ alignment: UtteranceAlignment, _ sequence: PhonemeSequence) -> Bool {
        let ids = [PhonemeVocabulary.silId] + alignment.phonemes.map { Int($0.phoneId) } + [PhonemeVocabulary.silId]
        return positions(of: sequence, in: ids) != nil
    }

    /// テキストから作った音素列の各音素が、アライメントの音素列（MAS が見つけた句読点の無い休止を含む）のどこにあるか
    ///
    /// ids: 文頭・文末の無音を含むアライメント側の音素 ID 列。テキスト側に無い音素はポーズだけを許す。合わなければ nil
    public static func positions(of sequence: PhonemeSequence, in ids: [Int]) -> [Int]? {
        var result: [Int] = []
        result.reserveCapacity(sequence.count)
        var i = 0
        for token in sequence.tokens {
            while i < ids.count && ids[i] != token.id && ids[i] == PhonemeVocabulary.pauId {
                i += 1
            }
            if ids.count <= i || ids[i] != token.id {
                return nil
            }
            result.append(i)
            i += 1
        }
        return i == ids.count ? result : nil
    }

    /// テキストから作った音素列に、アライメントで見つかった休止を入れる（休止の特徴は 0）。合わなければ nil
    public static func merge(_ sequence: PhonemeSequence, into ids: [Int]) -> PhonemeSequence? {
        guard let map = positions(of: sequence, in: ids) else {
            return nil
        }
        if map.count == ids.count {
            return sequence
        }
        let pause = PhonemeToken(id: PhonemeVocabulary.pauId, symbol: sequence.tokens.first { $0.id == PhonemeVocabulary.pauId }?.symbol ?? "<pau>", category: .pause)
        var tokens = [PhonemeToken](repeating: pause, count: ids.count)
        var features = [[Float]](repeating: [Float](repeating: 0.0, count: PhonemeSequenceBuilder.featureCount), count: ids.count)
        for (k, i) in map.enumerated() {
            tokens[i] = sequence.tokens[k]
            features[i] = sequence.features[k]
        }
        return PhonemeSequence(tokens: tokens, features: features, phraseStarts: sequence.phraseStarts.map { map[min($0, map.count - 1)] })
    }

    /// MAS に渡す音素列: 本体に、句読点の無い休止の候補（任意のポーズ）を入れる
    ///
    /// 候補は発話の頭（文頭の無音の余白を母音が吸わないように）と、ポーズの無いアクセント句の境目。
    static func alignmentTokens(_ sequence: PhonemeSequence) -> (tokens: [PhonemeToken], optional: [Bool]) {
        let body = bodyTokens(sequence)
        if body.isEmpty {
            return ([], [])
        }
        let pause = PhonemeToken(id: PhonemeVocabulary.pauId, symbol: "<pau>", category: .pause)
        let starts = Set(sequence.phraseStarts.map { $0 - 1 })
        var tokens: [PhonemeToken] = []
        var optional: [Bool] = []
        for (i, token) in body.enumerated() {
            let afterPause = 0 < i && body[i - 1].category == .pause
            if token.category != .pause && (i == 0 || (starts.contains(i) && afterPause != true)) {
                tokens.append(pause)
                optional.append(true)
            }
            tokens.append(token)
            optional.append(false)
        }
        return (tokens, optional)
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
            let (tokens, optional) = alignmentTokens(sequenceBuilder.build(text: u.text))
            if tokens.isEmpty {
                return
            }
            guard let pcm = Corpus.loadPCM(path: u.wavPath, reader: reader) else {
                return
            }
            let mel = melExtractor.extractLogMel(pcm: pcm)
            let totalFrames = max(1, mel.count)
            let boundaries = SpeechBoundaries.detect(pcm: pcm, totalFrames: totalFrames)
            if boundaries.speechFrames < optional.filter({ $0 != true }).count * 2 {
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
                phonemes: tokens,
                optional: optional
            )
            lock.lock()
            results[index] = item
            lock.unlock()
        }
        return results.compactMap { $0 }
    }
}
