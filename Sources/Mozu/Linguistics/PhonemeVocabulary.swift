import Foundation

/// 日本語 64 音素語彙テーブルおよび音素・モーラ相互変換機構
///
/// SNN 音響モデルの埋め込み層およびスパイク入力層のチャネル数を 64 に整流化し、
/// Apple Silicon SIMD や MLX テンソル演算時のメモリアライメント効率を最大化する。
public struct PhonemeVocabulary: Sendable {
    public static let padId: Int = 0
    public static let silId: Int = 1
    public static let unkId: Int = 2
    public static let sosId: Int = 3
    public static let eosId: Int = 4
    public static let qId: Int = 25
    public static let pauId: Int = 39

    public let size: Int
    private let tokenList: [String]
    private let tokenToIdTable: [String: Int]

    public init() {
        var list: [String] = []
        // 特殊トークン (0..4)
        list.append("<pad>") // 0: パディング
        list.append("<sil>") // 1: 文頭・文末・句点長無音 (Sentence Silence)
        list.append("<unk>") // 2: 未知音素
        list.append("<sos>") // 3: 発話開始記号
        list.append("<eos>") // 4: 発話終了記号

        // 日本語基本母音 (5..9)
        list.append("a") // 5
        list.append("i") // 6
        list.append("u") // 7
        list.append("e") // 8
        list.append("o") // 9

        // 日本語子音・特殊拍 (10..25)
        list.append("k") // 10
        list.append("s") // 11
        list.append("t") // 12
        list.append("n") // 13
        list.append("h") // 14
        list.append("m") // 15
        list.append("y") // 16
        list.append("r") // 17
        list.append("w") // 18
        list.append("g") // 19
        list.append("z") // 20
        list.append("d") // 21
        list.append("b") // 22
        list.append("p") // 23
        list.append("N") // 24 (撥音: ん)
        list.append("Q") // 25 (促音: っ)

        // 拡張・拗音・長音 (26..38)
        list.append("_")  // 26 (長音: ー)
        list.append("sh") // 27
        list.append("ch") // 28
        list.append("ts") // 29
        list.append("ky") // 30
        list.append("ny") // 31
        list.append("hy") // 32
        list.append("my") // 33
        list.append("ry") // 34
        list.append("gy") // 35
        list.append("j")  // 36
        list.append("by") // 37
        list.append("py") // 38

        // TTS 用ポーズトークン (39)
        list.append("<pau>") // 39 (読点休止: 、)

        // 予約トークン (40..63) 計64語彙
        var rIdx = 40
        while rIdx < 64 {
            list.append("<res\(rIdx)>")
            rIdx += 1
        }

        self.size = list.count
        self.tokenList = list

        var table: [String: Int] = [:]
        var i = 0
        while i < list.count {
            table[list[i]] = i
            i += 1
        }
        self.tokenToIdTable = table
    }

    /// トークン記号から音素 ID を取得する。
    /// 配列の線形探索を避け、定数時間ハッシュ検索により高速なトークン解決を行う。
    public func id(for token: String) -> Int {
        switch tokenToIdTable[token] {
        case .some(let val):
            return val
        case .none:
            return Self.unkId
        }
    }

    /// 音素 ID からトークン記号を取得する。
    /// SNN モデルからの不正インデックス出力によるクラッシュを未然に防止し、
    /// 範囲外の ID に対してはフォールバックとして未知音素トークンを返却する。
    public func token(for id: Int) -> String {
        switch true {
        case id < 0:
            return "<unk>"
        case tokenList.count <= id:
            return "<unk>"
        default:
            return tokenList[id]
        }
    }

    /// 音素カテゴリの判定。
    /// 音響モデルでの有声・無声マスクの自動生成および標準時間長の決定に必要なカテゴリを特定する。
    public func category(for symbol: String) -> PhonemeCategory {
        switch symbol {
        case "a", "i", "u", "e", "o":
            return .vowel
        case "N":
            return .nasalSyllable
        case "Q":
            return .geminate
        case "_":
            return .prolonged
        case "<sil>", "<pau>", "<pad>":
            return .pause
        case "ky", "ny", "hy", "my", "ry", "gy", "by", "py", "sh", "ch", "ts", "j":
            return .contracted
        default:
            return .consonant
        }
    }

    /// 音素が有声音（母音、有声子音、撥音、長音）であるか判定する。
    /// 無声子音区間およびポーズ区間において F0 ピッチ輪郭をゼロにマスクし、
    /// 有声音区間のみに自然なピッチ周波数と有声フラグを付与する。
    public func isVoiced(symbol: String) -> Bool {
        switch symbol {
        case "a", "i", "u", "e", "o", "N", "_":
            return true
        case "n", "m", "y", "r", "w", "g", "z", "d", "b":
            return true
        case "ny", "my", "ry", "gy", "j", "by":
            return true
        default:
            return false
        }
    }

    /// 音素 ID が完全無音・休止・促音（気流遮断）であるか判定する
    public func isPauseOrSilence(id: Int) -> Bool {
        switch id {
        case Self.silId, Self.pauId, Self.padId, Self.qId:
            return true
        default:
            return false
        }
    }

    /// 音素 ID が無声破裂音（k, t, p, ky, py）であるか判定する
    public func isUnvoicedStop(id: Int) -> Bool {
        switch id {
        case 10, 12, 23, 30, 38: // k, t, p, ky, py
            return true
        default:
            return false
        }
    }

    /// 音素 ID が有声破裂音（g, d, b, gy, by）であるか判定する
    public func isVoicedStop(id: Int) -> Bool {
        switch id {
        case 19, 21, 22, 35, 37: // g, d, b, gy, by
            return true
        default:
            return false
        }
    }

    /// 音素 ID が破擦音（ch, ts）であるか判定する
    public func isAffricate(id: Int) -> Bool {
        switch id {
        case 28, 29: // ch, ts
            return true
        default:
            return false
        }
    }

    /// 音素 ID が無声摩擦音（s, h, sh, hy）であるか判定する
    /// なぜ hy を含めるか:
    /// 「ひゃ」「ひゅ」「ひょ」の子音 hy は無声硬口蓋摩擦音 [ç] であり、
    /// 有声音ではなく無声摩擦気流としてモデル化する必要があるため。
    public func isUnvoicedFricative(id: Int) -> Bool {
        switch id {
        case 11, 14, 27, 32: // s, h, sh, hy
            return true
        default:
            return false
        }
    }

    /// 音素 ID が無声子音（無声破裂・無声摩擦・無声破擦）であるか判定する
    /// なぜ hy を含めるか:
    /// 音響モデルの無声子音コンテキスト判定およびボコーダー励起制御において、
    /// 硬口蓋摩擦音 hy を漏れなく無声子音として扱うため。
    public func isUnvoicedConsonant(id: Int) -> Bool {
        switch id {
        case 10, 11, 12, 14, 23, 27, 28, 29, 30, 32, 38: // k, s, t, h, p, sh, ch, ts, ky, hy, py
            return true
        default:
            return false
        }
    }

    /// 音素 ID が母音（a, i, u, e, o, 長音）であるか判定する
    public func isVowel(id: Int) -> Bool {
        switch id {
        case 5, 6, 7, 8, 9, 26: // a, i, u, e, o, _ (長音)
            return true
        default:
            return false
        }
    }

    /// 音素 ID が有声（母音・有声子音・撥音）であるか判定する
    public static func isVoicedPhone(phoneId: Int) -> Bool {
        switch phoneId {
        case 5, 6, 7, 8, 9, 26: // a, i, u, e, o, _
            return true
        case 13, 15, 16, 17, 18, 19, 20, 21, 22, 24: // n, m, y, r, w, g, z, d, b, N
            return true
        case 31, 33, 34, 35, 36, 37: // ny, my, ry, gy, j, by
            return true
        default:
            return false
        }
    }

    /// 音素列全体を見て各音素の有声/無声を判定する（母音の無声化規則を含む）
    ///
    /// なぜ文脈依存の無声化を入れるか:
    /// 日本語の /i/ /u/ は無声子音に挟まれた位置、または無声子音の後の句末（「です」「ます」の「す」）で
    /// 無声化する。音素単位の規則では常に有声扱いとなり、F0 と有声フラグが与えられたボコーダが
    /// 句末で低いブザー音を生成して「発話の後にロボットの残響が付く」ように聞こえていた。
    /// 学習時（F0 損失マスク・教師 F0）と推論時（ボコーダ入力）で同じ関数を使い整合させる。
    public static func contextVoicedFlags(phoneIds: [Int32]) -> [Bool] {
        let n = phoneIds.count
        var flags = [Bool](repeating: false, count: n)
        func isBoundary(_ pid: Int) -> Bool {
            return pid == silId || pid == pauId
        }
        func isVoicelessConsonant(_ pid: Int) -> Bool {
            if isBoundary(pid) {
                return false
            }
            if isVowelOrSpecialMora(phoneId: pid) {
                return false
            }
            return isVoicedPhone(phoneId: pid) != true
        }
        var i = 0
        while i < n {
            let pid = Int(phoneIds[i])
            var voiced = isVoicedPhone(phoneId: pid)
            if pid == 6 || pid == 7 { // i, u
                var prevVoiceless = false
                if 0 < i {
                    prevVoiceless = isVoicelessConsonant(Int(phoneIds[i - 1]))
                }
                var nextDevoicing = false
                if (i + 1) < n {
                    let next = Int(phoneIds[i + 1])
                    if isBoundary(next) || isVoicelessConsonant(next) {
                        nextDevoicing = true
                    }
                } else {
                    nextDevoicing = true
                }
                if prevVoiceless && nextDevoicing {
                    voiced = false
                }
            }
            flags[i] = voiced
            i += 1
        }
        return flags
    }

    /// 音素 ID が母音または特殊拍（促音・撥音・長音）であるか判定する
    public static func isVowelOrSpecialMora(phoneId: Int) -> Bool {
        switch phoneId {
        case 5, 6, 7, 8, 9, 24, 25, 26: // a, i, u, e, o, N, Q, _
            return true
        default:
            return false
        }
    }

    /// カタカナ文字をひらがなに正規化する。
    /// カタカナとひらがなは Unicode コードポイントが 0x60 オフセットで並行配置されており、
    /// 文字列置換ライブラリ呼び出しを排してスカラー演算のみで一括変換する。
    public func normalizeKatakanaToHiragana(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            let val = scalar.value
            switch true {
            case 0x30A1 <= val && val <= 0x30F6:
                if let hiraganaScalar = UnicodeScalar(val - 0x60) {
                    result.append(Character(hiraganaScalar))
                } else {
                    result.append(Character(scalar))
                }
            default:
                result.append(Character(scalar))
            }
        }
        return result
    }

    /// ひらがな・カタカナ表記から音素トークン系列を抽出する。
    /// 「き」と「ょ」のように個別に音素化すると日本語モーラ構造が崩れるため、
    /// 2文字の拗音表記を最長一致で優先検出し、正確な直交音素列を抽出する。
    public func kanaToPhonemes(_ text: String) -> [String] {
        let normalized = normalizeKatakanaToHiragana(text)
        let chars = Array(normalized)
        var phonemes: [String] = []
        var i = 0
        let count = chars.count

        while i < count {
            let c = chars[i]

            // 句読点・記号のポーズトークン変換
            switch c {
            case "、":
                phonemes.append("<pau>")
                i += 1
                continue
            case "。", "！", "？":
                phonemes.append("<sil>")
                i += 1
                continue
            case " ", "　":
                // 空白は短ポーズとする
                phonemes.append("<pau>")
                i += 1
                continue
            default:
                break
            }

            // 2文字の拗音・複合文字判定
            if (i + 1) < count {
                let nextC = chars[i + 1]
                let pair = String([c, nextC])
                var matched = true
                switch pair {
                case "きゃ": phonemes.append(contentsOf: ["ky", "a"])
                case "きゅ": phonemes.append(contentsOf: ["ky", "u"])
                case "きょ": phonemes.append(contentsOf: ["ky", "o"])
                case "しゃ": phonemes.append(contentsOf: ["sh", "a"])
                case "しゅ": phonemes.append(contentsOf: ["sh", "u"])
                case "しょ": phonemes.append(contentsOf: ["sh", "o"])
                case "しぇ": phonemes.append(contentsOf: ["sh", "e"])
                case "ちゃ": phonemes.append(contentsOf: ["ch", "a"])
                case "ちゅ": phonemes.append(contentsOf: ["ch", "u"])
                case "ちょ": phonemes.append(contentsOf: ["ch", "o"])
                case "ちぇ": phonemes.append(contentsOf: ["ch", "e"])
                case "にゃ": phonemes.append(contentsOf: ["ny", "a"])
                case "にゅ": phonemes.append(contentsOf: ["ny", "u"])
                case "にょ": phonemes.append(contentsOf: ["ny", "o"])
                case "ひゃ": phonemes.append(contentsOf: ["hy", "a"])
                case "ひゅ": phonemes.append(contentsOf: ["hy", "u"])
                case "ひょ": phonemes.append(contentsOf: ["hy", "o"])
                case "みゃ": phonemes.append(contentsOf: ["my", "a"])
                case "みゅ": phonemes.append(contentsOf: ["my", "u"])
                case "みょ": phonemes.append(contentsOf: ["my", "o"])
                case "りゃ": phonemes.append(contentsOf: ["ry", "a"])
                case "りゅ": phonemes.append(contentsOf: ["ry", "u"])
                case "りょ": phonemes.append(contentsOf: ["ry", "o"])
                case "ぎゃ": phonemes.append(contentsOf: ["gy", "a"])
                case "ぎゅ": phonemes.append(contentsOf: ["gy", "u"])
                case "ぎょ": phonemes.append(contentsOf: ["gy", "o"])
                case "じゃ": phonemes.append(contentsOf: ["j", "a"])
                case "じゅ": phonemes.append(contentsOf: ["j", "u"])
                case "じょ": phonemes.append(contentsOf: ["j", "o"])
                case "じぇ": phonemes.append(contentsOf: ["j", "e"])
                case "びゃ": phonemes.append(contentsOf: ["by", "a"])
                case "びゅ": phonemes.append(contentsOf: ["by", "u"])
                case "びょ": phonemes.append(contentsOf: ["by", "o"])
                case "ぴゃ": phonemes.append(contentsOf: ["py", "a"])
                case "ぴゅ": phonemes.append(contentsOf: ["py", "u"])
                case "ぴょ": phonemes.append(contentsOf: ["py", "o"])
                case "つぁ": phonemes.append(contentsOf: ["ts", "a"])
                case "つぃ": phonemes.append(contentsOf: ["ts", "i"])
                case "つぇ": phonemes.append(contentsOf: ["ts", "e"])
                case "つぉ": phonemes.append(contentsOf: ["ts", "o"])
                case "ふぁ": phonemes.append(contentsOf: ["h", "a"])
                case "ふぃ": phonemes.append(contentsOf: ["h", "i"])
                case "ふぇ": phonemes.append(contentsOf: ["h", "e"])
                case "ふぉ": phonemes.append(contentsOf: ["h", "o"])
                default:
                    matched = false
                }

                if matched {
                    i += 2
                    continue
                }
            }

            // 1文字の音素分解
            switch c {
            case "あ", "ぁ": phonemes.append("a")
            case "い", "ぃ": phonemes.append("i")
            case "う", "ぅ": phonemes.append("u")
            case "え", "ぇ": phonemes.append("e")
            case "お", "ぉ": phonemes.append("o")
            case "か": phonemes.append(contentsOf: ["k", "a"])
            case "き": phonemes.append(contentsOf: ["k", "i"])
            case "く": phonemes.append(contentsOf: ["k", "u"])
            case "け": phonemes.append(contentsOf: ["k", "e"])
            case "こ": phonemes.append(contentsOf: ["k", "o"])
            case "さ": phonemes.append(contentsOf: ["s", "a"])
            case "し": phonemes.append(contentsOf: ["sh", "i"])
            case "す": phonemes.append(contentsOf: ["s", "u"])
            case "せ": phonemes.append(contentsOf: ["s", "e"])
            case "そ": phonemes.append(contentsOf: ["s", "o"])
            case "た": phonemes.append(contentsOf: ["t", "a"])
            case "ち": phonemes.append(contentsOf: ["ch", "i"])
            case "つ": phonemes.append(contentsOf: ["ts", "u"])
            case "て": phonemes.append(contentsOf: ["t", "e"])
            case "と": phonemes.append(contentsOf: ["t", "o"])
            case "な": phonemes.append(contentsOf: ["n", "a"])
            case "に": phonemes.append(contentsOf: ["n", "i"])
            case "ぬ": phonemes.append(contentsOf: ["n", "u"])
            case "ね": phonemes.append(contentsOf: ["n", "e"])
            case "の": phonemes.append(contentsOf: ["n", "o"])
            case "は": phonemes.append(contentsOf: ["h", "a"])
            case "ひ": phonemes.append(contentsOf: ["h", "i"])
            case "ふ": phonemes.append(contentsOf: ["h", "u"])
            case "へ": phonemes.append(contentsOf: ["h", "e"])
            case "ほ": phonemes.append(contentsOf: ["h", "o"])
            case "ま": phonemes.append(contentsOf: ["m", "a"])
            case "み": phonemes.append(contentsOf: ["m", "i"])
            case "む": phonemes.append(contentsOf: ["m", "u"])
            case "め": phonemes.append(contentsOf: ["m", "e"])
            case "も": phonemes.append(contentsOf: ["m", "o"])
            case "や", "ゃ": phonemes.append(contentsOf: ["y", "a"])
            case "ゆ", "ゅ": phonemes.append(contentsOf: ["y", "u"])
            case "よ", "ょ": phonemes.append(contentsOf: ["y", "o"])
            case "ら": phonemes.append(contentsOf: ["r", "a"])
            case "り": phonemes.append(contentsOf: ["r", "i"])
            case "る": phonemes.append(contentsOf: ["r", "u"])
            case "れ": phonemes.append(contentsOf: ["r", "e"])
            case "ろ": phonemes.append(contentsOf: ["r", "o"])
            case "わ": phonemes.append(contentsOf: ["w", "a"])
            case "を": phonemes.append("o")
            case "ん": phonemes.append("N")
            case "っ": phonemes.append("Q")
            case "ー": phonemes.append("_")
            case "が": phonemes.append(contentsOf: ["g", "a"])
            case "ぎ": phonemes.append(contentsOf: ["gy", "i"])
            case "ぐ": phonemes.append(contentsOf: ["g", "u"])
            case "げ": phonemes.append(contentsOf: ["g", "e"])
            case "ご": phonemes.append(contentsOf: ["g", "o"])
            case "ざ": phonemes.append(contentsOf: ["z", "a"])
            case "じ": phonemes.append(contentsOf: ["j", "i"])
            case "ず": phonemes.append(contentsOf: ["z", "u"])
            case "ぜ": phonemes.append(contentsOf: ["z", "e"])
            case "ぞ": phonemes.append(contentsOf: ["z", "o"])
            case "だ": phonemes.append(contentsOf: ["d", "a"])
            case "ぢ": phonemes.append(contentsOf: ["j", "i"])
            case "づ": phonemes.append(contentsOf: ["z", "u"])
            case "で": phonemes.append(contentsOf: ["d", "e"])
            case "ど": phonemes.append(contentsOf: ["d", "o"])
            case "ば": phonemes.append(contentsOf: ["b", "a"])
            case "び": phonemes.append(contentsOf: ["b", "i"])
            case "ぶ": phonemes.append(contentsOf: ["b", "u"])
            case "べ": phonemes.append(contentsOf: ["b", "e"])
            case "ぼ": phonemes.append(contentsOf: ["b", "o"])
            case "ぱ": phonemes.append(contentsOf: ["p", "a"])
            case "ぴ": phonemes.append(contentsOf: ["p", "i"])
            case "ぷ": phonemes.append(contentsOf: ["p", "u"])
            case "ぺ": phonemes.append(contentsOf: ["p", "e"])
            case "ぽ": phonemes.append(contentsOf: ["p", "o"])
            default:
                // なぜフォールバック音素を追加するか:
                // かなテーブルに該当しない特殊文字であっても音素を脱落させず、
                // 確実に音声を合成するため。
                phonemes.append("a")
            }
            i += 1
        }
        return phonemes
    }

    /// ひらがな文字列をモーラ（拍）トークン列へ分解する。
    /// 日本語音声の韻律（アクセントトーン）はモーラに対して定義され、
    /// 音素への時間長配分もモーラ内部の子音・母音比率に基づいて決定されるため、モーラ単位に構造化する。
    public func kanaToMoras(_ text: String) -> [MoraToken] {
        let normalized = normalizeKatakanaToHiragana(text)
        let chars = Array(normalized)
        var moras: [MoraToken] = []
        var i = 0
        let count = chars.count

        while i < count {
            let c = chars[i]

            // 句読点
            switch c {
            case "、":
                let p = PhonemeToken(id: Self.pauId, symbol: "<pau>", category: .pause, durationFrames: 15)
                moras.append(MoraToken(text: "、", phonemes: [p], tone: .low, isAccentKernel: false, pitchF0: 0.0))
                i += 1
                continue
            case "。", "！", "？":
                let p = PhonemeToken(id: Self.silId, symbol: "<sil>", category: .pause, durationFrames: 30)
                moras.append(MoraToken(text: "。", phonemes: [p], tone: .low, isAccentKernel: false, pitchF0: 0.0))
                i += 1
                continue
            case " ", "　":
                let p = PhonemeToken(id: Self.pauId, symbol: "<pau>", category: .pause, durationFrames: 10)
                moras.append(MoraToken(text: " ", phonemes: [p], tone: .low, isAccentKernel: false, pitchF0: 0.0))
                i += 1
                continue
            default:
                break
            }

            // 2文字拗音（2文字目が小書き文字の場合のみ判定し、長音符「ー」や母音の誤合体を防止する）
            if (i + 1) < count {
                let nextC = chars[i + 1]
                var isSmallChar = false
                switch nextC {
                case "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゃ", "ゅ", "ょ", "ゎ",
                     "ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ヮ":
                    isSmallChar = true
                default:
                    break
                }

                if isSmallChar {
                    let pair = String([c, nextC])
                    let phonemeStrs = kanaToPhonemes(pair)
                    if phonemeStrs.count == 2 {
                        let pTokens = phonemeStrs.map { sym in
                            PhonemeToken(id: id(for: sym), symbol: sym, category: category(for: sym))
                        }
                        moras.append(MoraToken(text: pair, phonemes: pTokens))
                        i += 2
                        continue
                    }
                }
            }

            // 1文字モーラ
            let single = String(c)
            let phonemeStrs = kanaToPhonemes(single)
            if phonemeStrs.isEmpty != true {
                let pTokens = phonemeStrs.map { sym in
                    PhonemeToken(id: id(for: sym), symbol: sym, category: category(for: sym))
                }
                moras.append(MoraToken(text: single, phonemes: pTokens))
            } else {
                // なぜフォールバック母音を割り当てるか:
                // 完全に未知の記号や文字が混入した場合でも音素をスキップ（脱落）させず、
                // 発話音声の途中で音が消滅する不具合を物理的に防止するため。
                let sym = "a"
                let pToken = PhonemeToken(id: id(for: sym), symbol: sym, category: .vowel)
                moras.append(MoraToken(text: single, phonemes: [pToken]))
            }
            i += 1
        }

        return moras
    }
}
