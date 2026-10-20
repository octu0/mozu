import Foundation

/// 音声処理の基本定数
///
/// 言語処理フロントエンド、音響モデル、ボコーダの間でフレーム周期（10 ms）と Mel 次元を揃える。
public struct AudioConfig: Sendable {
    public static let sampleRate: Int = 16000
    public static let hopSize: Int = 160      // 10 ms
    public static let frameSize: Int = 320    // 20 ms（Mel 分析窓）
    public static let melChannels: Int = 64
}

/// 形態素解析における品詞分類
public enum PartOfSpeech: UInt8, Sendable, CaseIterable, Codable {
    case noun          = 0 // 名詞
    case verb          = 1 // 動詞
    case adjective     = 2 // 形容詞
    case adverb        = 3 // 副詞
    case particle      = 4 // 助詞
    case auxiliaryVerb = 5 // 助動詞
    case prefix        = 6 // 接頭辞
    case suffix        = 7 // 接尾辞
    case conjunction   = 8 // 接続詞
    case interjection  = 9 // 感嘆詞
    case symbol        = 10 // 記号
    case unknown       = 11 // 未知語
}

/// 音素の音響生理学的カテゴリ
public enum PhonemeCategory: UInt8, Sendable {
    case vowel         = 0 // 母音
    case consonant     = 1 // 子音
    case contracted    = 2 // 拗音および拡張子音
    case geminate      = 3 // 促音
    case nasalSyllable = 4 // 撥音
    case prolonged     = 5 // 長音
    case pause         = 6 // ポーズおよび無音
}

/// 東京方言アクセントの高低トーン
public enum AccentTone: UInt8, Sendable {
    case low  = 0
    case high = 1
}

/// 音素トークン
public struct PhonemeToken: Sendable, Equatable {
    public let id: Int
    public let symbol: String
    public let category: PhonemeCategory
    public var durationFrames: Int

    public init(id: Int, symbol: String, category: PhonemeCategory, durationFrames: Int = 0) {
        self.id = id
        self.symbol = symbol
        self.category = category
        self.durationFrames = durationFrames
    }
}

/// モーラ
///
/// 日本語のリズムとピッチアクセントはモーラ単位で決まるため、高低トーンと核の情報をモーラに持たせる。
public struct MoraToken: Sendable, Equatable {
    public let text: String
    public var phonemes: [PhonemeToken]
    public var tone: AccentTone
    public var isAccentKernel: Bool
    public var pitchF0: Float

    public init(
        text: String,
        phonemes: [PhonemeToken],
        tone: AccentTone = .low,
        isAccentKernel: Bool = false,
        pitchF0: Float = 0.0
    ) {
        self.text = text
        self.phonemes = phonemes
        self.tone = tone
        self.isAccentKernel = isAccentKernel
        self.pitchF0 = pitchF0
    }

    public var totalDurationFrames: Int {
        var sum = 0
        for p in phonemes {
            sum += p.durationFrames
        }
        return sum
    }
}

/// アクセント句
public struct AccentPhrase: Sendable, Equatable {
    public var moras: [MoraToken]
    public var pauseAfter: Bool
    public var pauseDurationFrames: Int
    public var isQuestion: Bool

    public init(
        moras: [MoraToken],
        pauseAfter: Bool = false,
        pauseDurationFrames: Int = 0,
        isQuestion: Bool = false
    ) {
        self.moras = moras
        self.pauseAfter = pauseAfter
        self.pauseDurationFrames = pauseDurationFrames
        self.isQuestion = isQuestion
    }
}

/// 話者・話し方の指定
///
/// 音響モデルは学習コーパスのラベル（話者、性別、年代、テンポ、テンション、方言）を条件として学習する。
/// `labels` に無い種類は「未指定」として扱われ、学習時にラベルを落とす訓練をしているため欠けていても合成できる。
public struct VoiceProfile: Sendable, Codable, Equatable {
    public let name: String
    /// ラベル種類 → 値（例: ["gender": "female", "dialect": "標準語"]）
    public var labels: [String: String]
    /// 話速倍率（1.0 で学習データどおり）
    public var speed: Float
    /// ピッチ倍率（1.0 で学習データどおり）
    public var pitch: Float
    /// 抑揚の倍率（1.0 で学習データどおり、大きいほど高低差が大きい）
    public var intonation: Float = 1.0
    /// 声の張り（エネルギーの移動、学習データの標準偏差単位。0 で学習データどおり）
    public var energy: Float = 0.0
    /// 話し方の潜在変数 z（話し方を持つ重みのみ。nil で平均的な話し方）
    public var latent: [Float]? = nil

    public init(name: String, labels: [String: String] = [:], speed: Float = 1.0, pitch: Float = 1.0) {
        self.name = name
        self.labels = labels
        self.speed = Self.clamp(speed, 0.5, 2.0, fallback: 1.0)
        self.pitch = Self.clamp(pitch, 0.5, 2.0, fallback: 1.0)
    }

    static func clamp(_ v: Float, _ lo: Float, _ hi: Float, fallback: Float) -> Float {
        if v.isFinite != true {
            return fallback
        }
        return min(hi, max(lo, v))
    }

    public static let female = VoiceProfile(name: "female", labels: ["gender": "female"])
    public static let male = VoiceProfile(name: "male", labels: ["gender": "male"])
    public static let neutral = VoiceProfile(name: "neutral")
    public static let `default` = female

    /// プリセット名または "key=value,key=value" 形式からプロファイルを作る
    public static func preset(named name: String) -> VoiceProfile {
        switch name.lowercased() {
        case "female", "woman":
            return .female
        case "male", "man":
            return .male
        case "neutral", "default", "":
            return .neutral
        default:
            break
        }
        var labels: [String: String] = [:]
        for pair in name.split(separator: ",") {
            let kv = pair.split(separator: "=", maxSplits: 1).map { String($0).trimmingCharacters(in: .whitespaces) }
            if kv.count == 2 && kv[0].isEmpty != true {
                labels[kv[0]] = kv[1]
            }
        }
        if labels.isEmpty {
            return .neutral
        }
        return VoiceProfile(name: name, labels: labels)
    }
}
