import Foundation

/// 日本語テキストの事前正規化・数詞展開・助詞置換・発音正規化パイプライン
///
/// 漢字かな混じり表記と実際の発音との乖離を整流化し、
/// SNN音響モデルへの入力音素を一貫させる。
public final class TextNormalizer: Sendable {
    private let morphology: ViterbiMorphology

    /// 語末の「う」を長音に変換してはならない終止形動詞セット
    ///
    /// 名詞の母音融合による長音化とは異なり、
    /// 動詞終止形の語尾は独立した母音拍として発音されるため長音化から保護する。
    public static let uEndingVerbs: Set<String> = [
        "思う", "追う", "問う", "酔う", "沿う", "食う", "吸う", "縫う", "乞う", "負う", "覆う", "請う",
        "装う", "添う", "集う", "揃う", "争う", "救う", "狂う", "通う", "見舞う", "住まう", "買う", "会う",
    ]

    /// 1桁数字のかな読みテーブル
    private static let digitKana: [String] = [
        "ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"
    ]

    public init(morphology: ViterbiMorphology = ViterbiMorphology()) {
        self.morphology = morphology
    }

    /// 入力テキストの全角記号・波ダッシュ正規化および注記除去
    ///
    /// 表記揺れを事前に統一しておくことで辞書のマッチング精度を向上させる。
    public func cleanText(_ text: String) -> String {
        if text.isEmpty {
            return ""
        }
        // macOS CLI やファイルシステム等から入力される NFD (結合文字) を NFC (正準等価事前合成) に統一
        let nfcText = text.precomposedStringWithCanonicalMapping
        let laughCleaned = removeLaughAnnotations(nfcText)
        let prolongedNormalized = normalizeProlongedMarks(laughCleaned)
        return prolongedNormalized
    }

    /// 括弧内の笑い注記等を除去する
    ///
    /// テキスト中の感情記号など発話対象外の注記を除去する。
    public func removeLaughAnnotations(_ text: String) -> String {
        let chars = Array(text)
        var out: [Character] = []
        out.reserveCapacity(chars.count)
        var i = 0
        let count = chars.count

        while i < count {
            let c = chars[i]
            if c == "(" || c == "（" {
                var j = i + 1
                var hasLaugh = false
                while j < count && j - i <= 7 && chars[j] != ")" && chars[j] != "）" {
                    if chars[j] == "笑" {
                        hasLaugh = true
                    }
                    j += 1
                }
                if hasLaugh && j < count && (chars[j] == ")" || chars[j] == "）") {
                    i = j + 1
                    continue
                }
            }
            out.append(c)
            i += 1
        }
        return String(out)
    }

    /// 波ダッシュ等の各種引き伸ばし記号を長音符「ー」に統一する
    ///
    /// 音素語彙テーブルの長音トークンに正確に対応させるため、各種記号を統一する。
    public func normalizeProlongedMarks(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            let val = scalar.value
            switch val {
            case 0x301C, // WAVE DASH (〜)
                 0xFF5E, // FULLWIDTH TILDE (～)
                 0x3030, // WAVY DASH (〰)
                 0x223C, // TILDE OPERATOR (∼)
                 0xFF70, // HALFWIDTH KATAKANA-HIRAGANA PROLONGED SOUND MARK (ｰ)
                 0x2015, // HORIZONTAL BAR (―)
                 0x2014, // EM DASH (—)
                 0x2013, // EN DASH (–)
                 0x2500, // BOX DRAWINGS LIGHT HORIZONTAL (─)
                 0xFF0D: // FULLWIDTH HYPHEN-MINUS (－)
                result.append("ー")
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    /// 算用数字および助数詞の組み合わせを自然な日本語読みに展開する
    ///
    /// 日本語の数字と単位の連接に伴う独自の音便規則を反映し、自然な発話を生成する。
    public func expandNumbersAndCounters(_ text: String) -> String {
        let chars = Array(text)
        let count = chars.count
        var result = ""
        var i = 0

        while i < count {
            let c = chars[i]
            let isDigit = isDigitChar(c)

            if isDigit {
                // 連続する数字列を抽出
                var numStr = ""
                while i < count && isDigitChar(chars[i]) {
                    numStr.append(chars[i])
                    i += 1
                }

                // 後続の助数詞を判定
                var counterStr = ""
                if i < count {
                    let nextC = chars[i]
                    switch true {
                    case i + 1 < count && chars[i] == "か" && chars[i + 1] == "月":
                        counterStr = "か月"
                        i += 2
                    case "本匹個年月日時候人冊枚階杯つ円分秒".contains(nextC):
                        counterStr = String(nextC)
                        i += 1
                    default:
                        break
                    }
                }

                if counterStr.isEmpty != true {
                    let expanded = expandNumeralWithCounter(numStr: numStr, counter: counterStr)
                    result.append(expanded)
                } else {
                    let numReading = numberReading(numStr)
                    result.append(numReading)
                }
                continue
            }

            result.append(c)
            i += 1
        }

        return result
    }

    /// 文字が数字（半角または全角）であるか判定
    private func isDigitChar(_ c: Character) -> Bool {
        let scalar = c.unicodeScalars.first!
        let v = scalar.value
        switch true {
        case 0x30 <= v && v <= 0x39:
            return true
        case 0xFF10 <= v && v <= 0xFF19:
            return true
        default:
            return false
        }
    }

    /// 数字文字列から整数値の桁配列を取得
    private func digitValues(_ text: String) -> [Int] {
        var values: [Int] = []
        for c in text.unicodeScalars {
            let v = c.value
            switch true {
            case 0x30 <= v && v <= 0x39:
                values.append(Int(v - 0x30))
            case 0xFF10 <= v && v <= 0xFF19:
                values.append(Int(v - 0xFF10))
            default:
                break
            }
        }
        return values
    }

    /// 0〜9999 の4桁位取り読み
    ///
    /// 日本語の位取り音便を正確に再現する。
    public func fourDigitReading(_ value: Int) -> String {
        var result = ""
        let thousands = value / 1000
        let hundreds = (value / 100) % 10
        let tens = (value / 10) % 10
        let ones = value % 10

        switch thousands {
        case 0:
            break
        case 1:
            result.append("せん")
        case 3:
            result.append("さんぜん")
        case 8:
            result.append("はっせん")
        default:
            result.append(Self.digitKana[thousands])
            result.append("せん")
        }

        switch hundreds {
        case 0:
            break
        case 1:
            result.append("ひゃく")
        case 3:
            result.append("さんびゃく")
        case 6:
            result.append("ろっぴゃく")
        case 8:
            result.append("はっぴゃく")
        default:
            result.append(Self.digitKana[hundreds])
            result.append("ひゃく")
        }

        switch tens {
        case 0:
            break
        case 1:
            result.append("じゅう")
        default:
            result.append(Self.digitKana[tens])
            result.append("じゅう")
        }

        if 0 < ones {
            result.append(Self.digitKana[ones])
        }

        return result
    }

    /// 数字列を万進法位取りでかな読みに展開する
    ///
    /// 日本語の数体系に基づき、4桁グループ単位で計算して自然な読みを生成する。
    public func numberReading(_ text: String) -> String {
        let values = digitValues(text)
        if values.isEmpty {
            return ""
        }

        // 先頭がゼロ（電話番号等）または17桁以上の長大数値は1桁ずつ読む
        if values[0] == 0 || 16 < values.count {
            if values.count == 1 {
                return "ぜろ"
            }
            var digitWise = ""
            for v in values {
                digitWise.append(Self.digitKana[v])
            }
            return digitWise
        }

        var number: Int64 = 0
        for v in values {
            number = number * 10 + Int64(v)
        }

        if number == 0 {
            return "ぜろ"
        }

        let groupUnits = ["", "まん", "おく", "ちょう"]
        var groups: [Int] = []
        var rest = number
        while 0 < rest {
            groups.append(Int(rest % 10000))
            rest /= 10000
        }

        var result = ""
        var gi = groups.count - 1
        while 0 <= gi {
            let g = groups[gi]
            if 0 < g {
                // 1万・1億は「いちまん」「いちおく」と読む
                if g == 1 && 0 < gi {
                    result.append("いち")
                } else {
                    result.append(fourDigitReading(g))
                }
                result.append(groupUnits[gi])
            }
            gi -= 1
        }

        return result
    }

    /// 数字と助数詞の組み合わせの特殊読みおよび連声展開
    ///
    /// 日付や人数などの不規則な読みの変化を確実に解決する。
    public func expandNumeralWithCounter(numStr: String, counter: String) -> String {
        let values = digitValues(numStr)
        var num = 0
        if values.count <= 4 {
            for v in values {
                num = num * 10 + v
            }
        }

        // 1. 日付・時刻・人数の不規則読み
        switch counter {
        case "日":
            switch num {
            case 1: return "ついたち"
            case 2: return "ふつか"
            case 3: return "みっか"
            case 4: return "よっか"
            case 5: return "いつか"
            case 6: return "むいか"
            case 7: return "なのか"
            case 8: return "ようか"
            case 9: return "ここのか"
            case 10: return "とおか"
            case 14: return "じゅうよっか"
            case 20: return "はつか"
            case 24: return "にじゅうよっか"
            default:
                return numberReading(numStr) + "にち"
            }
        case "月":
            switch num {
            case 4: return "しがつ"
            case 7: return "しちがつ"
            case 9: return "くがつ"
            default:
                return numberReading(numStr) + "がつ"
            }
        case "時":
            switch num {
            case 4: return "よじ"
            case 7: return "しちじ"
            case 9: return "くじ"
            default:
                return numberReading(numStr) + "じ"
            }
        case "人":
            switch num {
            case 1: return "ひとり"
            case 2: return "ふたり"
            case 4: return "よにん"
            default:
                return numberReading(numStr) + "にん"
            }
        case "つ":
            switch num {
            case 1: return "ひとつ"
            case 2: return "ふたつ"
            case 3: return "みっつ"
            case 4: return "よっつ"
            case 5: return "いつつ"
            case 6: return "むっつ"
            case 7: return "ななつ"
            case 8: return "やっつ"
            case 9: return "ここのつ"
            case 10: return "とお"
            default:
                return numberReading(numStr) + "こ"
            }
        case "本":
            // 促音化・半濁音化・濁音化 (いっぽん、さんぼん、じゅっぽん)
            switch num {
            case 1: return "いっぽん"
            case 2: return "にほん"
            case 3: return "さんぼん"
            case 4: return "よんほん"
            case 5: return "ごほん"
            case 6: return "ろっぽん"
            case 7: return "ななほん"
            case 8: return "はっぽん"
            case 9: return "きゅうほん"
            case 10: return "じゅっぽん"
            default:
                let base = numberReading(numStr)
                if base.hasSuffix("さん") || base.hasSuffix("せん") || base.hasSuffix("まん") {
                    return base + "ぼん"
                }
                if base.hasSuffix("いち") || base.hasSuffix("はち") || base.hasSuffix("じゅう") {
                    return String(base.dropLast()) + "っぽん"
                }
                return base + "ほん"
            }
        case "匹":
            switch num {
            case 1: return "いっぴき"
            case 2: return "にひき"
            case 3: return "さんびき"
            case 4: return "よんひき"
            case 5: return "ごひき"
            case 6: return "ろっぴき"
            case 7: return "ななひき"
            case 8: return "はっぴき"
            case 9: return "きゅうひき"
            case 10: return "じゅっぴき"
            default:
                return numberReading(numStr) + "ひき"
            }
        case "個":
            switch num {
            case 1: return "いっこ"
            case 6: return "ろっこ"
            case 8: return "はっこ"
            case 10: return "じゅっこ"
            default:
                return numberReading(numStr) + "こ"
            }
        case "年":
            return numberReading(numStr) + "ねん"
        case "分":
            switch num {
            case 1: return "いっぷん"
            case 3: return "さんぷん"
            case 4: return "よんぷん"
            case 6: return "ろっぷん"
            case 8: return "はっぷん"
            case 10: return "じゅっぷん"
            default:
                let base = numberReading(numStr)
                if base.hasSuffix("さん") || base.hasSuffix("よん") {
                    return base + "ぷん"
                }
                return base + "ふん"
            }
        case "秒":
            return numberReading(numStr) + "びょう"
        case "円":
            return numberReading(numStr) + "えん"
        case "冊":
            switch num {
            case 1: return "いっさつ"
            case 8: return "はっさつ"
            case 10: return "じゅっさつ"
            default:
                return numberReading(numStr) + "さつ"
            }
        case "枚":
            return numberReading(numStr) + "まい"
        case "階":
            switch num {
            case 1: return "いっかい"
            case 3: return "さんがい"
            case 6: return "ろっかい"
            case 8: return "はっかい"
            case 10: return "じゅっかい"
            default:
                return numberReading(numStr) + "かい"
            }
        case "杯":
            switch num {
            case 1: return "いっぱい"
            case 2: return "にはい"
            case 3: return "さんばい"
            case 6: return "ろっぱい"
            case 8: return "はっぱい"
            case 10: return "じゅっぱい"
            default:
                return numberReading(numStr) + "はい"
            }
        case "か月":
            switch num {
            case 1: return "いっかげつ"
            case 6: return "ろっかげつ"
            case 8: return "はっかげつ"
            case 10: return "じゅっかげつ"
            default:
                return numberReading(numStr) + "かげつ"
            }
        default:
            return numberReading(numStr) + counter
        }
    }

    /// かな文字の母音を取得する
    ///
    /// 母音連続の長音化判定および動詞終止形の保護を行う。
    private func vowel(of kana: Character) -> Character? {
        switch kana {
        case "あ", "か", "が", "さ", "ざ", "た", "だ", "な", "は", "ば", "ぱ", "ま", "や", "ら", "わ", "ぁ", "ゃ", "ゎ":
            return "あ"
        case "い", "き", "ぎ", "し", "じ", "ち", "ぢ", "に", "ひ", "び", "ぴ", "み", "り", "ゐ", "ぃ":
            return "い"
        case "う", "く", "ぐ", "す", "ず", "つ", "づ", "ぬ", "ふ", "ぶ", "ぷ", "む", "ゆ", "る", "ぅ", "ゅ", "ゔ":
            return "う"
        case "え", "け", "げ", "せ", "ぜ", "て", "で", "ね", "へ", "べ", "ぺ", "め", "れ", "ゑ", "ぇ":
            return "え"
        case "お", "こ", "ご", "そ", "ぞ", "と", "ど", "の", "ほ", "ぼ", "ぽ", "も", "よ", "ろ", "を", "ぉ", "ょ":
            return "お"
        default:
            return nil
        }
    }

    /// 発音正規化
    ///
    /// 現代日本語の発声規則に基づき、母音連続を長音表記に一致させて明瞭に発音させる。
    public func normalizePronunciation(surface: String, reading: String, pos: PartOfSpeech) -> String {
        // 助詞の読み替え
        if pos == .particle {
            switch surface {
            case "は": return "わ"
            case "へ": return "え"
            case "を": return "お"
            default: break
            }
        }

        // 慣用挨拶語の助詞 "は" の発音適正化
        switch surface {
        case "こんにちは":
            return "こんにちわ"
        case "こんばんは":
            return "こんばんわ"
        default:
            break
        }

        let chars = Array(reading)
        var result: [Character] = []
        result.reserveCapacity(chars.count)
        let keepsFinalU = Self.uEndingVerbs.contains(surface)
        var i = 0

        while i < chars.count {
            var c = chars[i]

            // ぢ/づの正規化
            switch c {
            case "ぢ": c = "じ"
            case "づ": c = "ず"
            default: break
            }

            // 直前母音との連続長音化判定
            if 0 < i, let prevVowel = vowel(of: result[result.count - 1]) {
                let isLast = (i == chars.count - 1)
                var lengthens = false
                switch c {
                case "う":
                    // お段+う, う段+う の長音化。ただし動詞終止形 uEndingVerbs は保護
                    lengthens = (prevVowel == "お" || prevVowel == "う") && (isLast && keepsFinalU) != true
                case "い":
                    // え段+い, い段+い の長音化
                    lengthens = (prevVowel == "え" || prevVowel == "い")
                case "え":
                    lengthens = (prevVowel == "え")
                case "お":
                    lengthens = (prevVowel == "お")
                case "あ":
                    lengthens = (prevVowel == "あ")
                default:
                    break
                }
                if lengthens {
                    c = "ー"
                }
            }

            result.append(c)
            i += 1
        }

        return String(result)
    }

    /// テキスト全文を形態素解析・発音正規化し、正規化済み形態素系列を出力する
    ///
    /// 事前整形、数詞展開、形態素解析、助詞置換、発音正規化を
    /// 順序正しく結合し、音声合成に最適化されたモーラ系列の土台を構築する。
    public func normalize(text: String) -> [Morpheme] {
        if text.isEmpty {
            return []
        }

        // 1. 事前文字整形
        let cleaned = cleanText(text)
        // 2. 数詞・助数詞の展開
        let expanded = expandNumbersAndCounters(cleaned)
        // 3. Viterbi 形態素解析
        let morphemes = morphology.tokenize(expanded)

        // 4. 文脈に応じた助詞置換と発音正規化
        var normalizedMorphemes: [Morpheme] = []
        var i = 0
        while i < morphemes.count {
            let m = morphemes[i]
            let normReading = normalizePronunciation(surface: m.surface, reading: m.reading, pos: m.pos)
            normalizedMorphemes.append(Morpheme(
                surface: m.surface,
                reading: normReading,
                pos: m.pos,
                accentKernel: m.accentKernel
            ))
            i += 1
        }

        return normalizedMorphemes
    }
}
