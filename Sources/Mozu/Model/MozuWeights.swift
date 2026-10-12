import Foundation

/// 条件付けラベルの語彙（話者・性別・年代・テンポ・テンション・方言など）
///
/// 値はコーパスのラベルから学習時に集め、重みに保存する。index 0 は「未指定」。
public struct LabelVocabulary: Sendable, Codable, Equatable {
    /// 種類の順序（音響モデルの埋め込みの順序と一致させる）
    public var kinds: [String]
    /// 種類 → 値の一覧（index 1 から）
    public var values: [String: [String]]

    public init(kinds: [String] = [], values: [String: [String]] = [:]) {
        self.kinds = kinds
        self.values = values
    }

    public func count(kind: String) -> Int {
        return (values[kind] ?? []).count + 1
    }

    /// 0 = 未指定
    public func index(kind: String, value: String?) -> Int {
        guard let v = value, let list = values[kind], let i = list.firstIndex(of: v) else {
            return 0
        }
        return i + 1
    }

    public func value(kind: String, index: Int) -> String? {
        guard let list = values[kind], 1 <= index, index <= list.count else {
            return nil
        }
        return list[index - 1]
    }

    /// 既存の値を保ったまま新しい値を追加する（学習済みの埋め込み index を変えないため）
    public mutating func register(kind: String, value: String) {
        if kinds.contains(kind) != true {
            kinds.append(kind)
        }
        var list = values[kind] ?? []
        if list.contains(value) != true {
            list.append(value)
            values[kind] = list
        }
    }

    /// VoiceProfile のラベルを index 列に変換する
    public func indices(for labels: [String: String]) -> [Int] {
        return kinds.map { index(kind: $0, value: labels[$0]) }
    }
}

/// mozu の重みコンテナ
///
/// 語彙（表記→読み・アクセント核）は辞書としてソースに置かず、ここに蓄積する。
/// 音響モデルとボコーダは別々に学習・差し替えできるように任意にしてある。
public struct MozuWeights: Sendable, Codable {
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var lexicon: [LexiconEntry]
    public var labels: LabelVocabulary
    public var acoustic: AcousticWeights?
    public var vocoder: VocoderWeights?

    public init(lexicon: [LexiconEntry] = [], labels: LabelVocabulary = LabelVocabulary(), acoustic: AcousticWeights? = nil, vocoder: VocoderWeights? = nil) {
        self.formatVersion = Self.currentFormatVersion
        self.lexicon = lexicon
        self.labels = labels
        self.acoustic = acoustic
        self.vocoder = vocoder
    }

    public static func load(from url: URL) throws -> MozuWeights {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MozuWeights.self, from: data)
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// 旧形式（spikespeech）や別の重みファイルから語彙だけを取り込む
    public static func loadLexicon(from url: URL) throws -> [LexiconEntry] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(LexiconCarrier.self, from: data).lexicon
    }

    public var parameterCount: Int {
        return (acoustic?.parameterCount ?? 0) + (vocoder?.parameterCount ?? 0)
    }
}
