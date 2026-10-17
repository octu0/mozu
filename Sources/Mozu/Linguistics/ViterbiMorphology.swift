import Foundation

/// 形態素解析辞書エントリ
///
/// クラスによる参照オーバーヘッドを排し、
/// ソート済み連続配列としてメモリ上に配置してキャッシュ効率を高める。
public struct LexiconEntry: Sendable, Equatable, Codable {
    public let surface: String
    public let reading: String
    public let pos: PartOfSpeech
    public let accentKernel: Int16
    public let cost: Int16

    public init(
        surface: String,
        reading: String,
        pos: PartOfSpeech,
        accentKernel: Int16 = 0,
        cost: Int16 = 100
    ) {
        self.surface = surface
        self.reading = reading
        self.pos = pos
        self.accentKernel = accentKernel
        self.cost = cost
    }
}

/// 形態素解析結果トークン
///
/// 後続の助詞置換判定および東京方言アクセント高低トーン付与に必要な属性を保持する。
public struct Morpheme: Sendable, Equatable {
    public let surface: String
    public let reading: String
    public let pos: PartOfSpeech
    public let accentKernel: Int16

    public init(surface: String, reading: String, pos: PartOfSpeech, accentKernel: Int16) {
        self.surface = surface
        self.reading = reading
        self.pos = pos
        self.accentKernel = accentKernel
    }
}

/// ラティス展開ノード
///
/// 動的計画法により各ノードに到達する最小コスト経路を線形時間で決定し、
/// バックポインタで最短パスを復元する。
public struct LatticeNode: Sendable {
    public let id: Int
    public let startPos: Int
    public let length: Int
    public let surface: String
    public let reading: String
    public let pos: PartOfSpeech
    public let accentKernel: Int16
    public let nodeCost: Int16
    public var cumulativeCost: Int
    public var bestPrevId: Int

    public init(
        id: Int,
        startPos: Int,
        length: Int,
        surface: String,
        reading: String,
        pos: PartOfSpeech,
        accentKernel: Int16,
        nodeCost: Int16,
        cumulativeCost: Int = Int.max / 2,
        bestPrevId: Int = -1
    ) {
        self.id = id
        self.startPos = startPos
        self.length = length
        self.surface = surface
        self.reading = reading
        self.pos = pos
        self.accentKernel = accentKernel
        self.nodeCost = nodeCost
        self.cumulativeCost = cumulativeCost
        self.bestPrevId = bestPrevId
    }
}

/// 未知語漢字音訳結果の並行安全なキャッシュストレージ
/// なぜキャッシュを導入するか:
/// 長文や同一熟語・漢字が反復されるテキストにおいて、CFStringTokenizer の生成オーバーヘッドを排し、
/// 11,000文字以上の大量テキスト正規化を 0.2 秒未満の高速な線形時間 O(N) で完了させるため。
private final class KanjiReadingCache: @unchecked Sendable {
    static let shared = KanjiReadingCache()
    private var cache: [String: String] = [:]
    private let lock = NSLock()

    func get(_ key: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]
    }

    func set(_ key: String, value: String) {
        lock.lock()
        defer { lock.unlock() }
        cache[key] = value
    }
}

/// Pure Swift 二分探索ラティス構築および品詞間遷移 Viterbi 最短経路形態素解析器
///
/// 外部環境への依存を排除し完全な並行安全性を満たしつつ、
/// 音声合成特化の語彙とアクセント核を直接統合した高精度な形態素分割を行う。
public final class ViterbiMorphology: Sendable {
    private let sortedLexicon: [LexiconEntry]
    private let transitionMatrix: [[Int16]]

    public init(lexicon: [LexiconEntry]? = nil) {
        let entries: [LexiconEntry]
        switch lexicon {
        case .some(let custom):
            if custom.isEmpty != true {
                entries = custom
            } else {
                entries = Self.loadDefaultLexicon()
            }
        case .none:
            entries = Self.loadDefaultLexicon()
        }
        // 文字列の共通接頭辞検索を二分探索で実行するため、辞書を表記の昇順で整列する。
        self.sortedLexicon = entries.sorted { a, b in
            a.surface < b.surface
        }
        self.transitionMatrix = Self.buildTransitionMatrix()
    }

    /// モデル重みファイルから永続化語彙をロードする
    /// なぜソースコード内ハードコードではなくモデルファイルから読み込むか:
    /// ソースコード内に静的辞書を残さず、獲得・学習された語彙知識をモデルの重みと
    /// 完全に一体化して管理・更新可能にするため。
    public static func loadDefaultLexicon() -> [LexiconEntry] {
        let baseList = buildFallbackLexicon()
        var mergedMap: [String: LexiconEntry] = [:]
        var bIdx = 0
        while bIdx < baseList.count {
            let item = baseList[bIdx]
            let key = "\(item.surface)_\(item.pos.rawValue)"
            mergedMap[key] = item
            bIdx += 1
        }

        var candidates: [String] = []
        switch ProcessInfo.processInfo.environment["WEIGHTS_PATH"] {
        case .some(let envPath):
            if envPath.isEmpty != true {
                candidates.append(envPath)
            }
        case .none:
            break
        }
        candidates.append("Models/weights.json")
        candidates.append("../Models/weights.json")
        candidates.append("/app/Models/weights.json")
        candidates.append("Models/weights_test.json")
        candidates.append("../Models/weights_test.json")

        let fileManager = FileManager.default
        var i = 0
        while i < candidates.count {
            let path = candidates[i]
            if fileManager.fileExists(atPath: path) {
                let url = URL(fileURLWithPath: path)
                switch try? Data(contentsOf: url) {
                case .some(let data):
                    let decoder = JSONDecoder()
                    switch try? decoder.decode(LexiconCarrier.self, from: data) {
                    case .some(let weights):
                        if weights.lexicon.isEmpty != true {
                            var wIdx = 0
                            while wIdx < weights.lexicon.count {
                                let entry = weights.lexicon[wIdx]
                                let normEntry = LexiconEntry(
                                    surface: entry.surface.precomposedStringWithCanonicalMapping,
                                    reading: entry.reading.precomposedStringWithCanonicalMapping,
                                    pos: entry.pos,
                                    accentKernel: entry.accentKernel,
                                    cost: entry.cost
                                )
                                let key = "\(normEntry.surface)_\(normEntry.pos.rawValue)"
                                mergedMap[key] = normEntry
                                wIdx += 1
                            }
                            return Array(mergedMap.values)
                        }
                    case .none:
                        break
                    }
                case .none:
                    break
                }
            }
            i += 1
        }
        return Array(mergedMap.values)
    }

    /// 語彙獲得結果
    public struct LexiconLearningResult: Sendable {
        /// 既存語彙に獲得語彙を統合した語彙リスト
        public let lexicon: [LexiconEntry]
        /// 新規に追加されたエントリ数
        public let added: Int
        /// 獲得元の文数
        public let sentenceCount: Int
    }

    /// 獲得語彙に与えるコスト
    /// なぜ手書き基本語彙（5〜10）より高く未知語熟語ノード（200×文字数）より低くするか:
    /// 助詞・助動詞の手書き読み（は→わ 等）を獲得語彙で上書きせず、
    /// 一方で文全体の文脈から推定した読みが単漢字フォールバックより常に優先されるようにするため。
    public static let learnedLexiconCost: Int16 = 20

    /// 文の集合から語彙（表記→読み）を獲得し、既存語彙へ統合する
    ///
    /// なぜコーパス文を文全体として音訳してから語彙化するか:
    /// 漢字列だけを切り出して音訳すると送り仮名や文脈が失われ（溺れ→にれ、入れない→にゅーれない）、
    /// 文全体を日本語ロケールの形態素音訳に掛けることで読みの精度が大きく向上するため。
    /// 獲得した語彙はソースコードではなくモデル重み（MozuWeights.lexicon）に永続化される。
    ///
    /// 規則:
    /// - 漢字または「々」を含む語と、2 文字以上のカタカナ語・英字の語を獲得する（ひらがな語は既存語彙と未知語処理に任せ、助詞の読みを保護する）
    ///   カタカナ語・英字の語も獲得するのは、語彙に無い語にはアクセント学習の結果を記録できず、平板のままになるため
    /// - 英字の語の読みは綴りの規則による推定（EnglishReading.bestGuess）
    /// - readings（コーパスが与える表記 → 読み）にある語はその読みを使い、獲得済みの語の読みも置き換える
    /// - 連続する漢字含有語の連結（複合語）も獲得し、Viterbi で最長一致が優先されるようにする
    /// - 同一表記に複数の読みがある場合は出現頻度の高い読みを採用する
    /// - 既存語彙および基本語彙に同一表記（同一品詞）がある場合は上書きしない
    public static func learnLexiconEntries(
        sentences: [String],
        existing: [LexiconEntry],
        readings: [String: String] = [:]
    ) -> LexiconLearningResult {
        // 既存語彙が空の場合は基本語彙（助詞・助動詞など）を土台にする。
        // 獲得語彙だけの語彙リストを作ると、は→わ などの手書き読みが失われるため。
        var baseList = existing
        if baseList.isEmpty {
            baseList = buildFallbackLexicon()
        }
        var reserved = Set<String>()
        for entry in baseList {
            reserved.insert(entry.surface.precomposedStringWithCanonicalMapping)
        }
        var reservedFallback = Set<String>()
        for entry in buildFallbackLexicon() {
            reserved.insert(entry.surface.precomposedStringWithCanonicalMapping)
            reservedFallback.insert(entry.surface.precomposedStringWithCanonicalMapping)
        }

        var readingCounts: [String: [String: Int]] = [:]

        func containsKanji(_ s: String) -> Bool {
            for ch in s {
                if isKanji(ch) || ch == "々" {
                    return true
                }
            }
            return false
        }

        func isKatakanaWord(_ s: String) -> Bool {
            if s.count < 2 {
                return false
            }
            for ch in s {
                if isKatakanaOrProlonged(ch) != true || ch == "・" {
                    return false
                }
            }
            return true
        }

        func isLatinWord(_ s: String) -> Bool {
            return 2 <= s.count && s.allSatisfy { EnglishReading.isLatinLetter($0) }
        }

        func isHiraganaReading(_ s: String) -> Bool {
            if s.isEmpty {
                return false
            }
            for scalar in s.unicodeScalars {
                let v = scalar.value
                let isHira = (0x3041 <= v && v <= 0x3096)
                let isProlonged = (v == 0x30FC)
                if isHira != true && isProlonged != true {
                    return false
                }
            }
            return true
        }

        func record(surface: String, reading: String) {
            let normSurface = surface.precomposedStringWithCanonicalMapping
            let normReading = reading.precomposedStringWithCanonicalMapping
            if reserved.contains(normSurface) {
                return
            }
            if containsKanji(normSurface) != true && isKatakanaWord(normSurface) != true && isLatinWord(normSurface) != true {
                return
            }
            if isHiraganaReading(normReading) != true {
                return
            }
            var counts = readingCounts[normSurface] ?? [:]
            counts[normReading] = (counts[normReading] ?? 0) + 1
            readingCounts[normSurface] = counts
        }

        var sentenceCount = 0
        for sentence in sentences {
            // 英字の語は合成時と同じ正規化（全角→半角、語間の空白を詰める）をしてから取り出す
            for word in EnglishReading.latinRuns(in: EnglishReading.normalizeLatin(sentence.precomposedStringWithCanonicalMapping)) {
                if readings[word] == nil {
                    record(surface: word, reading: EnglishReading.bestGuess(word))
                }
            }
            let tokens = transcribeSentenceTokens(sentence)
            if tokens.isEmpty {
                continue
            }
            sentenceCount += 1

            var runSurface = ""
            var runReading = ""
            var runLength = 0
            var tIdx = 0
            while tIdx < tokens.count {
                let tok = tokens[tIdx]
                if containsKanji(tok.surface) {
                    record(surface: tok.surface, reading: tok.reading)
                    // 直後のかな語（送り仮名・活用語尾）を連結した表記も獲得する。
                    // なぜ連結するか: 悪/わる + さ、良/よ + い のように単漢字の読みが文脈で変わる場合に、
                    // 表記「悪さ」「良い」を 1 語として持たせ、頻度多数決による誤読（悪→あく）を避けるため。
                    // 助詞・助動詞・記号は連結しない（は→わ などの助詞発音規則を壊さないため）。
                    if (tIdx + 1) < tokens.count {
                        let nextTok = tokens[tIdx + 1]
                        if containsKanji(nextTok.surface) != true && isHiraganaReading(nextTok.surface) {
                            switch inferPartOfSpeech(for: nextTok.surface) {
                            case .particle, .auxiliaryVerb, .symbol:
                                break
                            default:
                                record(surface: tok.surface + nextTok.surface, reading: tok.reading + nextTok.reading)
                            }
                        }
                    }
                    runSurface += tok.surface
                    runReading += tok.reading
                    runLength += 1
                } else {
                    if isKatakanaWord(tok.surface) {
                        record(surface: tok.surface, reading: katakanaToHiragana(tok.surface))
                    }
                    if 1 < runLength {
                        record(surface: runSurface, reading: runReading)
                    }
                    runSurface = ""
                    runReading = ""
                    runLength = 0
                }
                tIdx += 1
            }
            if 1 < runLength {
                record(surface: runSurface, reading: runReading)
            }
        }

        var merged = baseList
        var added = 0
        // コーパスが与える読み: 獲得済みの語は読みを置き換え（読みが変われば核は未知に戻す）、無い語は追加する
        var given: [String: String] = [:]
        for (surface, reading) in readings {
            let normSurface = EnglishReading.normalizeLatin(surface.precomposedStringWithCanonicalMapping)
            let normReading = katakanaToHiragana(reading.precomposedStringWithCanonicalMapping)
            if isHiraganaReading(normReading) && reservedFallback.contains(normSurface) != true {
                given[normSurface] = normReading
                readingCounts[normSurface] = nil
            }
        }
        var index = 0
        while index < merged.count {
            let entry = merged[index]
            if let reading = given[entry.surface], entry.cost == learnedLexiconCost {
                if entry.reading != reading {
                    merged[index] = LexiconEntry(surface: entry.surface, reading: reading, pos: entry.pos, accentKernel: 0, cost: learnedLexiconCost)
                }
                given[entry.surface] = nil
            }
            index += 1
        }
        for surface in given.keys.sorted() where reserved.contains(surface) != true {
            merged.append(LexiconEntry(surface: surface, reading: given[surface]!, pos: .noun, accentKernel: 0, cost: learnedLexiconCost))
            added += 1
        }
        let surfaces = readingCounts.keys.sorted()
        for surface in surfaces {
            guard let counts = readingCounts[surface] else {
                continue
            }
            var bestReading = ""
            var bestCount = 0
            for (reading, count) in counts.sorted(by: { $0.key < $1.key }) {
                if bestCount < count {
                    bestCount = count
                    bestReading = reading
                }
            }
            if bestReading.isEmpty {
                continue
            }
            merged.append(LexiconEntry(
                surface: surface,
                reading: bestReading,
                pos: .noun,
                accentKernel: 0,
                cost: learnedLexiconCost
            ))
            added += 1
        }

        return LexiconLearningResult(lexicon: merged, added: added, sentenceCount: sentenceCount)
    }

    /// 文全体を日本語ロケールの単語単位で分かち書きし、各語の読み（ひらがな）を返す
    /// 読みが取得できない語は表記をそのまま読みとして返す。
    public static func transcribeSentenceTokens(_ sentence: String) -> [(surface: String, reading: String)] {
        var tokens: [(surface: String, reading: String)] = []
        #if canImport(CoreFoundation)
        let locale = Locale(identifier: "ja_JP") as CFLocale
        let nsText = sentence as NSString
        let cfText = sentence as CFString
        let tokenizer = CFStringTokenizerCreate(
            kCFAllocatorDefault,
            cfText,
            CFRangeMake(0, nsText.length),
            kCFStringTokenizerUnitWord,
            locale
        )
        var tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while tokenType.isEmpty != true {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let surface = nsText.substring(with: NSRange(location: range.location, length: range.length))
            var reading = surface
            switch CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, kCFStringTokenizerAttributeLatinTranscription) {
            case .some(let attr):
                if let latin = attr as? String {
                    let mutable = NSMutableString(string: latin)
                    if CFStringTransform(mutable as CFMutableString, nil, kCFStringTransformLatinHiragana, false) {
                        reading = mutable as String
                    }
                }
            case .none:
                break
            }
            tokens.append((surface: surface, reading: reading))
            tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        #endif
        return tokens
    }

    /// 品詞間接続コストを取得する
    ///
    /// 名詞の後に助詞が続くような自然な結合に低コストを与え、
    /// 記号の後に助詞が直結するような文法的不整合を高コストで排除する。
    public func transitionCost(from: PartOfSpeech, to: PartOfSpeech) -> Int16 {
        let r = Int(from.rawValue)
        let c = Int(to.rawValue)
        return transitionMatrix[r][c]
    }

    /// ソート済み語彙配列から指定の表層文字列に完全一致するエントリ群を二分探索する
    ///
    /// 辞書エントリが表記昇順で整列されているため、二分探索で先頭位置を特定し、
    /// 同一表記の異品詞エントリを含めて漏れなく連続区間から抽出する。
    public func findEntries(surface: String) -> [LexiconEntry] {
        var low = 0
        var high = sortedLexicon.count
        while low < high {
            let mid = low + (high - low) / 2
            if sortedLexicon[mid].surface < surface {
                low = mid + 1
            } else {
                high = mid
            }
        }
        var results: [LexiconEntry] = []
        var idx = low
        while idx < sortedLexicon.count && sortedLexicon[idx].surface == surface {
            results.append(sortedLexicon[idx])
            idx += 1
        }
        return results
    }

    /// テキストの指定開始位置から共通接頭辞検索を行う
    ///
    /// 最大語長までの各部分文字列に対して二分探索を適用し、確実に全一致エントリを抽出する。
    /// 語彙に無い外来語（カタカナ語・英字の語）のアクセント核
    ///
    /// 外来語の一般的な規則: 後ろから 3 拍目に核を置き、その拍が長音・撥音・促音なら 1 拍前にずらす。2 拍以下は頭高。
    static func loanwordAccentKernel(reading: String) -> Int16 {
        let small: Set<Character> = ["ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゎ"]
        var moras: [Character] = []
        for ch in reading where small.contains(ch) != true {
            moras.append(ch)
        }
        if moras.count <= 2 {
            return moras.isEmpty ? 0 : 1
        }
        var kernel = moras.count - 2
        if 1 < kernel && ["ー", "ん", "っ"].contains(moras[kernel - 1]) {
            kernel -= 1
        }
        return Int16(kernel)
    }

    /// 語彙照合で試す表記の最大文字数（長いカタカナ語・英字の語も一致させる）
    static let maxSurfaceLength = 20

    public func commonPrefixSearch(in chars: [Character], startOffset: Int) -> [LexiconEntry] {
        let count = chars.count
        if count <= startOffset {
            return []
        }

        var matches: [LexiconEntry] = []
        var maxLen = Self.maxSurfaceLength
        let remaining = count - startOffset
        if remaining < maxLen {
            maxLen = remaining
        }

        var len = 1
        while len <= maxLen {
            let sub = String(chars[startOffset ..< (startOffset + len)])
            let found = findEntries(surface: sub)
            matches.append(contentsOf: found)
            len += 1
        }

        return matches
    }

    /// テキストを形態素解析し、最適形態素系列を抽出する
    ///
    /// 可能な単語分割経路の数は入力長に対して増大するため、
    /// 各文字境界での最小累積コストノードのみを更新して多項式時間で最適解を導出する。
    public func tokenize(_ text: String) -> [Morpheme] {
        if text.isEmpty {
            return []
        }

        let chars = Array(text)
        let charCount = chars.count

        // 各文字位置で始まるノードのリストを管理 (0..charCount)
        var nodesStartingAt: [[LatticeNode]] = Array(repeating: [], count: charCount + 1)
        var nextNodeId = 1
        // 1. ラティスノードの構築
        var pos = 0
        while pos < charCount {
            let char = chars[pos]
            let matches = commonPrefixSearch(in: chars, startOffset: pos)

            for entry in matches {
                let len = entry.surface.count
                if 0 < len && pos + len <= charCount {
                    let node = LatticeNode(
                        id: nextNodeId,
                        startPos: pos,
                        length: len,
                        surface: entry.surface,
                        reading: entry.reading,
                        pos: entry.pos,
                        accentKernel: entry.accentKernel,
                        nodeCost: entry.cost
                    )
                    nodesStartingAt[pos].append(node)
                    nextNodeId += 1
                }
            }

            // 連続するカタカナ列（外来語・固有名詞）の検出と一括形態素ノード生成
            if Self.isKatakanaOrProlonged(char) {
                var kEnd = pos + 1
                while kEnd < charCount {
                    if Self.isKatakanaOrProlonged(chars[kEnd]) != true {
                        break
                    }
                    kEnd += 1
                }
                let kLen = kEnd - pos
                if 1 < kLen {
                    let katakanaStr = String(chars[pos..<kEnd])
                    let katakanaReading = Self.katakanaToHiragana(katakanaStr)
                    let katakanaNode = LatticeNode(
                        id: nextNodeId,
                        startPos: pos,
                        length: kLen,
                        surface: katakanaStr,
                        reading: katakanaReading,
                        pos: .noun,
                        accentKernel: Self.loanwordAccentKernel(reading: katakanaReading),
                        nodeCost: Int16(min(Int(Int16.max), 120 * kLen))
                    )
                    nodesStartingAt[pos].append(katakanaNode)
                    nextNodeId += 1
                }
            }

            // 連続する英字列（英単語・略語）の一括ノード。語彙に無い語を 1 文字ずつのアルファベット読みにしないため
            if EnglishReading.isLatinLetter(char) {
                var lEnd = pos + 1
                while lEnd < charCount && EnglishReading.isLatinLetter(chars[lEnd]) {
                    lEnd += 1
                }
                let lLen = lEnd - pos
                if 1 < lLen {
                    let word = String(chars[pos..<lEnd])
                    let latinNode = LatticeNode(
                        id: nextNodeId,
                        startPos: pos,
                        length: lLen,
                        surface: word,
                        reading: EnglishReading.bestGuess(word),
                        pos: .noun,
                        accentKernel: Self.loanwordAccentKernel(reading: EnglishReading.bestGuess(word)),
                        nodeCost: Int16(min(Int(Int16.max), 120 * lLen))
                    )
                    nodesStartingAt[pos].append(latinNode)
                    nextNodeId += 1
                }
            }

            // 連続する漢字列（熟語）の検出と一括音訳ノード生成
            if Self.isKanji(char) {
                var kEnd = pos + 1
                while kEnd < charCount {
                    if Self.isKanji(chars[kEnd]) != true {
                        break
                    }
                    kEnd += 1
                }
                let kLen = kEnd - pos
                if 1 < kLen {
                    let compoundStr = String(chars[pos..<kEnd])
                    let compoundReading = Self.fallbackReadingForKanji(compoundStr)
                    let compoundNode = LatticeNode(
                        id: nextNodeId,
                        startPos: pos,
                        length: kLen,
                        surface: compoundStr,
                        reading: compoundReading,
                        pos: .noun,
                        accentKernel: 0,
                        nodeCost: Int16(min(Int(Int16.max), 200 * kLen))
                    )
                    nodesStartingAt[pos].append(compoundNode)
                    nextNodeId += 1
                }
            }

            // 1文字未知語ノードを常に生成（ラティス上の未到達孤立点を防止）
            let (unknownPos, unknownReading, unknownCost) = determineUnknownCharProperties(char)
            let singleStr = String(char)
            let singleCost = Int(unknownCost) + 100
            let node = LatticeNode(
                id: nextNodeId,
                startPos: pos,
                length: 1,
                surface: singleStr,
                reading: unknownReading,
                pos: unknownPos,
                accentKernel: 0,
                nodeCost: Int16(min(Int(Int16.max), singleCost))
            )
            nodesStartingAt[pos].append(node)
            nextNodeId += 1

            pos += 1
        }

        // 仮想的な BOS (文頭) ノード
        let bosNode = LatticeNode(
            id: 0,
            startPos: -1,
            length: 0,
            surface: "<BOS>",
            reading: "",
            pos: .symbol,
            accentKernel: 0,
            nodeCost: 0,
            cumulativeCost: 0,
            bestPrevId: -1
        )

        // 2. Viterbi 前向きコスト積算
        // 位置 0 のノード群は BOS から接続
        var p0Idx = 0
        while p0Idx < nodesStartingAt[0].count {
            let nodeCost = Int(nodesStartingAt[0][p0Idx].nodeCost)
            let trans = Int(transitionCost(from: bosNode.pos, to: nodesStartingAt[0][p0Idx].pos))
            nodesStartingAt[0][p0Idx].cumulativeCost = nodeCost + trans
            nodesStartingAt[0][p0Idx].bestPrevId = 0
            p0Idx += 1
        }

        // 全位置の DP 更新
        var allNodesMap: [Int: LatticeNode] = [0: bosNode]
        var startPos = 0
        while startPos < charCount {
            var nIdx = 0
            while nIdx < nodesStartingAt[startPos].count {
                let currentNode = nodesStartingAt[startPos][nIdx]
                allNodesMap[currentNode.id] = currentNode

                let endPos = startPos + currentNode.length
                if endPos <= charCount {
                    // endPos で始まる後続ノード群への遷移を更新
                    var succIdx = 0
                    while succIdx < nodesStartingAt[endPos].count {
                        let succ = nodesStartingAt[endPos][succIdx]
                        let trans = Int(transitionCost(from: currentNode.pos, to: succ.pos))
                        let totalCost = currentNode.cumulativeCost + Int(succ.nodeCost) + trans
                        if totalCost < succ.cumulativeCost {
                            nodesStartingAt[endPos][succIdx].cumulativeCost = totalCost
                            nodesStartingAt[endPos][succIdx].bestPrevId = currentNode.id
                        }
                        succIdx += 1
                    }
                }
                nIdx += 1
            }
            startPos += 1
        }

        // 3. 文末 (EOS) に接続する最適ノードの選定
        var bestLastNode: LatticeNode? = nil
        var minEosCost = Int.max

        // 末尾文字に達した全ノードを走査
        var sPos = 0
        while sPos <= charCount {
            var nIdx = 0
            while nIdx < nodesStartingAt[sPos].count {
                let node = nodesStartingAt[sPos][nIdx]
                allNodesMap[node.id] = node
                if node.startPos + node.length == charCount {
                    let trans = Int(transitionCost(from: node.pos, to: .symbol))
                    let costWithEos = node.cumulativeCost + trans
                    if costWithEos < minEosCost {
                        minEosCost = costWithEos
                        bestLastNode = node
                    }
                }
                nIdx += 1
            }
            sPos += 1
        }

        guard var curr = bestLastNode else {
            return []
        }

        // 4. バックトラックによる最適形態素系列の復元
        var reversedMorphemes: [Morpheme] = []
        while 0 < curr.id {
            reversedMorphemes.append(Morpheme(
                surface: curr.surface,
                reading: curr.reading,
                pos: curr.pos,
                accentKernel: curr.accentKernel
            ))

            let prevId = curr.bestPrevId
            if prevId <= 0 {
                break
            }
            guard let prevNode = allNodesMap[prevId] else {
                break
            }
            curr = prevNode
        }

        return reversedMorphemes.reversed()
    }

    /// 辞書未登録の漢字文字・熟語に対する日本語音訓フォールバック読みを導出する
    /// なぜ CFStringTokenizer の日本語 LatinTranscription と音訳変換を用いるか:
    /// 単なる ICU kCFStringTransformToLatin は中国語ピンイン（水→shui、本→ben）に変換されてしまうため、
    /// 日本語ロケール（ja_JP）の形態素音訳属性を用いて正しい日本語の読み（水→みず、檸檬→れもん）を抽出し、
    private static let commonKanjiReadings: [Character: String] = [
        "好": "す", "日": "にち", "本": "ほん", "語": "ご", "入": "にゅう",
        "力": "りょく", "出": "しゅつ", "生": "せい", "成": "せい", "音": "おん",
        "声": "せい", "合": "ごう", "気": "き", "天": "てん", "晴": "せい",
        "今": "いま", "時": "じ", "間": "かん", "人": "ひと", "物": "もの",
        "事": "こと", "言": "こと", "話": "はなし", "大": "だい", "小": "しょう",
        "中": "ちゅう", "高": "こう", "低": "てい", "新": "しん", "古": "こ",
        "白": "しろ", "黒": "くろ", "赤": "あか", "青": "あお", "長": "ちょう",
        "短": "たん", "正": "せい", "誤": "ご", "真": "ま", "実": "じつ",
        "学": "がく", "校": "こう", "先": "せん", "名": "な", "前": "まえ",
        "後": "ご", "手": "て", "足": "あし", "目": "め", "耳": "みみ",
        "口": "くち", "心": "こころ", "道": "みち", "国": "くに", "山": "やま",
        "川": "かわ", "海": "うみ", "空": "そら", "雨": "あめ", "雪": "ゆき",
        "風": "かぜ", "木": "き", "林": "はやし", "森": "もり", "花": "はな",
        "草": "くさ", "車": "くるま", "電": "でん", "線": "せん", "駅": "えき",
        "店": "みせ", "買": "かい", "売": "うり", "読": "よみ", "書": "かき",
        "聞": "きき", "見": "み", "食": "たべ", "飲": "のみ", "行": "ゆき",
        "来": "き", "帰": "かえり", "待": "まち", "持": "もち", "知": "しり",
        "思": "おもい", "考": "かんがえ", "感": "かん", "情": "じょう", "愛": "あい",
        "友": "とも", "達": "たち", "親": "おや", "子": "こ", "男": "おとこ",
        "女": "おんな", "父": "ちち", "母": "はは", "兄": "あに", "弟": "おとうと",
        "姉": "あね", "妹": "いもうと", "家": "いえ", "屋": "や", "室": "しつ",
        "場": "ば", "所": "ところ", "方": "かた", "向": "むき", "上": "うえ",
        "下": "した", "左": "ひだり", "右": "みぎ", "東": "ひがし", "西": "にし",
        "南": "みなみ", "北": "きた"
    ]

    /// 音素脱落や誤音訳を防止するため。
    public static func fallbackReadingForKanji(_ text: String) -> String {
        switch KanjiReadingCache.shared.get(text) {
        case .some(let cached):
            return cached
        case .none:
            break
        }

        if text.count == 1 {
            switch text.first {
            case .some(let firstChar):
                switch commonKanjiReadings[firstChar] {
                case .some(let reading):
                    KanjiReadingCache.shared.set(text, value: reading)
                    return reading
                case .none:
                    break
                }
            case .none:
                break
            }
        }

        var resolved: String = ""
        #if canImport(CoreFoundation)
        let locale = Locale(identifier: "ja_JP") as CFLocale
        let cfText = text as CFString
        let textLen = (text as NSString).length
        let tokenizer = CFStringTokenizerCreate(
            kCFAllocatorDefault,
            cfText,
            CFRangeMake(0, textLen),
            kCFStringTokenizerUnitWordBoundary,
            locale
        )
        var result = ""
        var tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while tokenType.isEmpty != true {
            switch CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, kCFStringTokenizerAttributeLatinTranscription) {
            case .some(let attr):
                if let latin = attr as? String {
                    let mutable = NSMutableString(string: latin)
                    if CFStringTransform(mutable as CFMutableString, nil, kCFStringTransformLatinHiragana, false) {
                        result.append(mutable as String)
                    } else {
                        result.append(latin)
                    }
                }
            case .none:
                break
            }
            tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        if result.isEmpty != true && result != text {
            resolved = result
        }
        #endif

        if resolved.isEmpty {
            // フォールバック: 一般音訳変換
            let mutable = NSMutableString(string: text)
            let cfStr = mutable as CFMutableString
            if CFStringTransform(cfStr, nil, kCFStringTransformToLatin, false) {
                if CFStringTransform(cfStr, nil, kCFStringTransformStripDiacritics, false) {
                    if CFStringTransform(cfStr, nil, kCFStringTransformLatinHiragana, false) {
                        let converted = mutable as String
                        if converted.isEmpty != true && converted != text {
                            resolved = converted
                        }
                    }
                }
            }
        }

        if resolved.isEmpty {
            resolved = "あ"
        }

        KanjiReadingCache.shared.set(text, value: resolved)
        return resolved
    }

    /// 対象文字が漢字であるか判定する
    public static func isKanji(_ char: Character) -> Bool {
        switch char.unicodeScalars.first {
        case .some(let s):
            let v = s.value
            if 0x4E00 <= v && v <= 0x9FFF {
                return true
            }
            if 0x3400 <= v && v <= 0x4DBF {
                return true
            }
            if 0xF900 <= v && v <= 0xFAFF {
                return true
            }
            return false
        case .none:
            return false
        }
    }

    /// 対象文字がカタカナまたは長音符であるか判定する
    public static func isKatakanaOrProlonged(_ char: Character) -> Bool {
        switch char.unicodeScalars.first {
        case .some(let s):
            let v = s.value
            if 0x30A0 <= v && v <= 0x30FF {
                return true
            }
            if v == 0x30FC || v == 0xFF70 {
                return true
            }
            if 0xFF65 <= v && v <= 0xFF9F {
                return true
            }
            return false
        case .none:
            return false
        }
    }

    /// カタカナ文字列をひらがな文字列に変換する（長音記号は保持）
    public static func katakanaToHiragana(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            let v = scalar.value
            if 0x30A1 <= v && v <= 0x30F6 {
                switch UnicodeScalar(v - 0x60) {
                case .some(let hira):
                    result.unicodeScalars.append(hira)
                    continue
                case .none:
                    break
                }
            }
            result.unicodeScalars.append(scalar)
        }
        return result
    }

    /// 単語表記から文脈品詞を推定する
    public static func inferPartOfSpeech(for sub: String) -> PartOfSpeech {
        switch sub {
        case "、", "。", "！", "？", "…", "・":
            return .symbol
        case "は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで", "より", "か", "ね", "よ", "な", "わ", "て", "ば":
            return .particle
        case "だ", "です", "た", "ます", "ない", "たい", "らしい", "でした", "ません", "ました":
            return .auxiliaryVerb
        case "まだ", "また", "とても", "いつも", "少し", "やはり", "もう":
            return .adverb
        case "ある", "よる", "なら", "買わ":
            return .verb
        default:
            if sub.hasSuffix("ない") || sub.hasSuffix("なく") {
                return .auxiliaryVerb
            }
            if 1 < sub.count && sub.hasSuffix("い") {
                return .adjective
            }
            return .noun
        }
    }

    /// 未知文字の文字種に応じた品詞・読み・コストを決定する
    ///
    /// 辞書未登録の文字が出現しても解析を途絶させず、適切なフォールバック読みを付与する。
    private func determineUnknownCharProperties(_ char: Character) -> (PartOfSpeech, String, Int16) {
        let str = String(char)
        let scalar = char.unicodeScalars.first!
        let val = scalar.value

        switch val {
        case 0x3041...0x3096: // ひらがな
            return (.noun, str, 200)
        case 0x30A1...0x30FA: // カタカナ
            switch UnicodeScalar(val - 0x60) {
            case .some(let hiraScalar):
                return (.noun, String(Character(hiraScalar)), 200)
            case .none:
                return (.noun, str, 200)
            }
        case 0x30FC, 0xFF70: // 長音記号（ー）
            return (.noun, "ー", 150)
        case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0xF900...0xFAFF: // 漢字
            let kanjiReading = Self.fallbackReadingForKanji(str)
            return (.noun, kanjiReading, 400)
        case 0x30...0x39, 0xFF10...0xFF19: // 数字
            return (.noun, str, 150)
        case 0x41...0x5A, 0x61...0x7A, 0xFF21...0xFF3A, 0xFF41...0xFF5A: // 英字
            return (.noun, EnglishReading.letterName(char) ?? str, 300)
        default:
            return (.symbol, str, 250)
        }
    }

    /// 品詞間接続コスト行列を構築する
    ///
    /// 固定サイズで行列を事前展開し、探索ループ内での条件分岐を排除して定数時間で接続コストを評価する。
    private static func buildTransitionMatrix() -> [[Int16]] {
        let count = PartOfSpeech.allCases.count
        var matrix = Array(repeating: Array(repeating: Int16(50), count: count), count: count)

        // 名詞 -> 助詞 (強い結合)
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 0
        // 名詞 -> 助動詞 (だ、です)
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 10
        // 名詞 -> 接尾辞
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.suffix.rawValue)] = 0
        // 名詞 -> 動詞
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 30
        // 名詞 -> 記号
        matrix[Int(PartOfSpeech.noun.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 10

        // 動詞 -> 助動詞 (た、ます)
        matrix[Int(PartOfSpeech.verb.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 0
        // 動詞 -> 助詞 (て、に)
        matrix[Int(PartOfSpeech.verb.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 10
        // 動詞 -> 記号
        matrix[Int(PartOfSpeech.verb.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 10

        // 形容詞 -> 助動詞 / 助詞
        matrix[Int(PartOfSpeech.adjective.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 0
        matrix[Int(PartOfSpeech.adjective.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 10

        // 接頭辞 -> 名詞
        matrix[Int(PartOfSpeech.prefix.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 0

        // 助詞 -> 助詞 (の、は、など)
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 20
        // 助詞 -> 名詞
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 10
        // 助詞 -> 動詞
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 10
        // 助詞 -> 副詞 (は -> まだ, も -> 少し)
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.adverb.rawValue)] = 10
        // 助詞 -> 記号
        matrix[Int(PartOfSpeech.particle.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 0

        // 副詞 -> 形容詞 (まだ -> 無い, とても -> 良い)
        matrix[Int(PartOfSpeech.adverb.rawValue)][Int(PartOfSpeech.adjective.rawValue)] = 10
        // 副詞 -> 動詞 (まだ -> 走る)
        matrix[Int(PartOfSpeech.adverb.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 10
        // 副詞 -> 助動詞 (まだ -> だ)
        matrix[Int(PartOfSpeech.adverb.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 10
        // 副詞 -> 名詞 (少し -> 水)
        matrix[Int(PartOfSpeech.adverb.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 10

        // 助動詞 -> 助詞 (ですね、でしたか)
        matrix[Int(PartOfSpeech.auxiliaryVerb.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 10
        // 助動詞 -> 助動詞 (でした)
        matrix[Int(PartOfSpeech.auxiliaryVerb.rawValue)][Int(PartOfSpeech.auxiliaryVerb.rawValue)] = 10
        // 助動詞 -> 記号
        matrix[Int(PartOfSpeech.auxiliaryVerb.rawValue)][Int(PartOfSpeech.symbol.rawValue)] = 0

        // 記号 -> 助詞 (文法的不自然)
        matrix[Int(PartOfSpeech.symbol.rawValue)][Int(PartOfSpeech.particle.rawValue)] = 500
        // 記号 -> 名詞 / 動詞 (文頭や句点後の新文開始)
        matrix[Int(PartOfSpeech.symbol.rawValue)][Int(PartOfSpeech.noun.rawValue)] = 10
        matrix[Int(PartOfSpeech.symbol.rawValue)][Int(PartOfSpeech.verb.rawValue)] = 20

        return matrix
    }

    /// 外部モデルファイル不在時および未知語フォールバック用の基本語彙テーブル
    private static func buildFallbackLexicon() -> [LexiconEntry] {
        var list: [LexiconEntry] = []

        // 助詞
        let particles: [(String, String, Int16)] = [
            ("は", "わ", 0), ("が", "が", 0), ("を", "お", 0), ("に", "に", 0), ("へ", "え", 0),
            ("で", "で", 0), ("と", "と", 0), ("も", "も", 0), ("の", "の", 0), ("から", "から", 0),
            ("まで", "まで", 0), ("より", "より", 0), ("か", "か", 0), ("ね", "ね", 0), ("よ", "よ", 0),
            ("な", "な", 0), ("わ", "わ", 0), ("ぞ", "ぞ", 0), ("ぜ", "ぜ", 0), ("など", "など", 0),
            ("だけ", "だけ", 0), ("ほど", "ほど", 0), ("ばかり", "ばかり", 0), ("でも", "でも", 0)
        ]
        var pIdx = 0
        while pIdx < particles.count {
            let item = particles[pIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .particle,
                accentKernel: item.2,
                cost: 5
            ))
            pIdx += 1
        }

        // 助動詞
        let auxVerbs: [(String, String, Int16)] = [
            ("だ", "だ", 0), ("です", "です", 1), ("た", "た", 0), ("ます", "ます", 1),
            ("ない", "ない", 1), ("たい", "たい", 1), ("らしい", "らしい", 2),
            ("でした", "でした", 1), ("ません", "ません", 2), ("ました", "ました", 2),
            ("ください", "ください", 0), ("下さい", "ください", 0), ("なり", "なり", 1)
        ]
        var aIdx = 0
        while aIdx < auxVerbs.count {
            let item = auxVerbs[aIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .auxiliaryVerb,
                accentKernel: item.2,
                cost: 5
            ))
            aIdx += 1
        }

        // 代名詞・指示詞
        let pronouns: [(String, String, Int16)] = [
            ("これ", "これ", 0), ("それ", "それ", 0), ("あれ", "あれ", 0), ("どれ", "どれ", 1),
            ("ここ", "ここ", 0), ("そこ", "そこ", 0), ("あそこ", "あそこ", 0), ("どこ", "どこ", 1),
            ("私", "わたし", 0), ("わたし", "わたし", 0), ("僕", "ぼく", 1), ("ぼく", "ぼく", 1),
            ("彼", "かれ", 1), ("彼女", "かのじょ", 1), ("誰", "だれ", 1), ("何", "なに", 1)
        ]
        var prIdx = 0
        while prIdx < pronouns.count {
            let item = pronouns[prIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .noun,
                accentKernel: item.2,
                cost: 10
            ))
            prIdx += 1
        }

        // 基本名詞
        let nouns: [(String, String, Int16)] = [
            ("水", "みず", 0), ("花", "はな", 2), ("本", "ほん", 1), ("先生", "せんせー", 3),
            ("東京", "とーきょー", 0), ("日本", "にほん", 2), ("日本語", "にほんご", 0), ("靴", "くつ", 2),
            ("映画", "えーが", 0), ("猫", "ねこ", 1), ("ねこ", "ねこ", 1), ("桜", "さくら", 0),
            ("さくら", "さくら", 0), ("卵", "たまご", 2), ("学校", "がっこー", 0), ("雨", "あめ", 1),
            ("人", "ひと", 2), ("年", "ねん", 1), ("月", "つき", 2), ("日", "ひ", 0),
            ("名前", "なまえ", 0), ("吾輩", "わがはい", 0), ("音声", "おんせー", 0), ("合成", "ごーせー", 0),
            ("音声合成", "おんせーごーせー", 4), ("スパイク", "すぱいく", 0), ("スピーチ", "すぴーち", 0),
            ("スパイクスピーチ", "すぱいくすぴーち", 0), ("マレーシア", "まれーしあ", 0),
            ("テスト", "てすと", 1), ("テキスト", "てきすと", 1), ("入力", "にゅうりょく", 0),
            ("出力", "しゅつりょく", 0), ("お好き", "おすき", 2), ("本日", "ほんじつ", 1),
            ("晴天", "せいてん", 0), ("天気", "てんき", 1), ("今日", "きょう", 1),
            ("明日", "あした", 3), ("昨日", "きのう", 2), ("世界", "せかい", 1),
            ("人間", "にんげん", 0), ("情報", "じょうほう", 0), ("システム", "しすてむ", 1),
            ("データ", "でーた", 1), ("モデル", "もでる", 1), ("技術", "ぎじゅつ", 1),
            ("機能", "きのう", 1), ("処理", "しょり", 1), ("時間", "じかん", 0)
        ]
        var nIdx = 0
        while nIdx < nouns.count {
            let item = nouns[nIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .noun,
                accentKernel: item.2,
                cost: 20
            ))
            nIdx += 1
        }

        // 基本動詞
        let verbs: [(String, String, Int16)] = [
            ("思う", "おもう", 2), ("買う", "かう", 0), ("買わ", "かわ", 0),
            ("ある", "ある", 1), ("行く", "いく", 0), ("来る", "くる", 1),
            ("する", "する", 0), ("して", "して", 0), ("でき", "でき", 1),
            ("できる", "できる", 2), ("見る", "みる", 1), ("聞く", "きく", 0),
            ("読む", "よむ", 1), ("書く", "かく", 1), ("話す", "はなす", 2),
            ("走る", "はしる", 2), ("食べる", "たべる", 2), ("飲む", "のむ", 1),
            ("よる", "よる", 1), ("なら", "なら", 1)
        ]
        var vIdx = 0
        while vIdx < verbs.count {
            let item = verbs[vIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .verb,
                accentKernel: item.2,
                cost: 20
            ))
            vIdx += 1
        }

        // 形容詞
        let adjectives: [(String, String, Int16)] = [
            ("好き", "すき", 2), ("すき", "すき", 2), ("良い", "よい", 1), ("いい", "いい", 1),
            ("高い", "たかい", 2), ("低い", "ひくい", 2), ("新しい", "あたらしい", 4),
            ("古い", "ふるい", 2), ("美しい", "うつくしい", 4), ("自然", "しぜん", 0),
            ("必要", "ひつよう", 0), ("可能", "かのう", 0), ("重要", "じゅうよう", 0)
        ]
        var adjIdx = 0
        while adjIdx < adjectives.count {
            let item = adjectives[adjIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .adjective,
                accentKernel: item.2,
                cost: 15
            ))
            adjIdx += 1
        }

        // 挨拶・副詞・接続詞
        let phrases: [(String, String, PartOfSpeech, Int16)] = [
            ("こんにちは", "こんにちわ", .interjection, 0),
            ("こんばんは", "こんばんわ", .interjection, 0),
            ("おはよう", "おはよー", .interjection, 2),
            ("ありがとう", "ありがとー", .interjection, 2),
            ("さようなら", "さよーなら", .interjection, 3),
            ("はい", "はい", .interjection, 1),
            ("いいえ", "いいえ", .interjection, 3),
            ("そして", "そして", .conjunction, 0),
            ("しかし", "しかし", .conjunction, 2),
            ("また", "また", .conjunction, 0),
            ("とても", "とても", .adverb, 0),
            ("いつも", "いつも", .adverb, 1),
            ("少し", "すこし", .adverb, 2),
            ("まだ", "まだ", .adverb, 1)
        ]
        var phIdx = 0
        while phIdx < phrases.count {
            let item = phrases[phIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: item.2,
                accentKernel: item.3,
                cost: 15
            ))
            phIdx += 1
        }

        // 記号
        let symbols: [(String, String)] = [
            ("、", "<pau>"), ("。", "<sil>"), ("！", "<sil>"), ("？", "<sil>"),
            ("・", "<pau>"), ("…", "<pau>"), ("〜", "ー"), ("～", "ー")
        ]
        var sIdx = 0
        while sIdx < symbols.count {
            let item = symbols[sIdx]
            list.append(LexiconEntry(
                surface: item.0.precomposedStringWithCanonicalMapping,
                reading: item.1.precomposedStringWithCanonicalMapping,
                pos: .symbol,
                accentKernel: 0,
                cost: 5
            ))
            sIdx += 1
        }

        return list
    }
}

/// 重みファイルから語彙だけを読むための最小のデコード対象
///
/// 旧形式（spikespeech の重み）も新形式（MozuWeights）も `lexicon` キーを持つため、どちらからでも語彙を引き継げる。
public struct LexiconCarrier: Codable, Sendable {
    public let lexicon: [LexiconEntry]

    public init(lexicon: [LexiconEntry]) {
        self.lexicon = lexicon
    }
}
