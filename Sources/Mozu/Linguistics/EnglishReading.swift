import Foundation

/// 英字の語の日本語英語読み（カタカナ英語）
///
/// 単語の一覧は持たず、英語の綴りの一般的な規則（マジック e、母音字の組み合わせ、r 化母音、促音化など）だけで読む。
/// 転写に現れた語は bestGuess の読みで語彙（重み）に入り、アクセントは音声から学習される。
public enum EnglishReading {
    /// 全角英字を半角に直し、英字に挟まれた空白・ハイフン・「!」「:」「'」を詰める（「Snow Man」→「SnowMan」、「K-POP」→「KPOP」）
    ///
    /// なぜ詰めるか: これらは記号として句を区切りポーズになるが、英字の名前や複合語の中では続けて 1 語として読むため。
    public static func normalizeLatin(_ text: String) -> String {
        var chars: [Character] = []
        chars.reserveCapacity(text.count)
        for ch in text {
            if let v = ch.unicodeScalars.first?.value, ch.unicodeScalars.count == 1, (0xFF21 <= v && v <= 0xFF3A) || (0xFF41 <= v && v <= 0xFF5A) {
                chars.append(Character(UnicodeScalar(v - 0xFEE0)!))
            } else {
                chars.append(ch)
            }
        }
        var out = ""
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            if joiners.contains(ch) {
                var j = i
                while j < chars.count && joiners.contains(chars[j]) {
                    j += 1
                }
                if 0 < i && j < chars.count && isLatinLetter(chars[i - 1]) && isLatinLetter(chars[j]) {
                    i = j
                    continue
                }
            }
            out.append(ch)
            i += 1
        }
        return out
    }

    /// 英字に挟まれていれば詰める文字
    static let joiners: Set<Character> = [" ", "\u{3000}", "-", "‐", "－", "!", "！", ":", "：", "'", "’"]

    /// 半角英字か
    public static func isLatinLetter(_ ch: Character) -> Bool {
        guard let v = ch.unicodeScalars.first?.value, ch.unicodeScalars.count == 1 else {
            return false
        }
        return (0x41 <= v && v <= 0x5A) || (0x61 <= v && v <= 0x7A)
    }

    /// 文中の英字の連なり（normalizeLatin 済みの文に対して）
    public static func latinRuns(in text: String) -> [String] {
        var runs: [String] = []
        var current = ""
        for ch in text {
            if isLatinLetter(ch) {
                current.append(ch)
            } else if current.isEmpty != true {
                runs.append(current)
                current = ""
            }
        }
        if current.isEmpty != true {
            runs.append(current)
        }
        return runs
    }

    /// 読み（ひらがな）。大文字だけの短い語や母音字の無い語は略語としてアルファベット読みにし、
    /// 先頭に大文字が続いてから小文字になる語（VTuber）は、最後の大文字の手前までをアルファベット読みにする
    public static func bestGuess(_ word: String) -> String {
        if looksLikeAcronym(word) {
            return spelled(word)
        }
        let letters = Array(word)
        var upper = 0
        while upper < letters.count && letters[upper].isUppercase {
            upper += 1
        }
        if 2 <= upper && upper < letters.count {
            let head = String(letters[0..<(upper - 1)])
            let tail = String(letters[(upper - 1)...])
            return spelled(head) + (phonics(tail).first ?? spelled(tail))
        }
        return phonics(word).first ?? spelled(word)
    }

    /// 略語らしいか（大文字だけで 3 文字以下、または母音字を含まない）
    static func looksLikeAcronym(_ word: String) -> Bool {
        let letters = Array(word)
        if letters.count <= 1 {
            return true
        }
        let upper = letters.allSatisfy { $0.isUppercase }
        if upper && letters.count <= 3 {
            return true
        }
        let vowels = letters.contains { "aeiouyAEIOUY".contains($0) }
        return vowels != true
    }

    /// アルファベット読み（「SNS」→「えすえぬえす」）
    public static func spelled(_ word: String) -> String {
        var out = ""
        for ch in word {
            out += letterName(ch) ?? ""
        }
        return out
    }

    /// 英字 1 文字のアルファベット読み
    public static func letterName(_ ch: Character) -> String? {
        switch ch.lowercased() {
        case "a": return "えー"
        case "b": return "びー"
        case "c": return "しー"
        case "d": return "でぃー"
        case "e": return "いー"
        case "f": return "えふ"
        case "g": return "じー"
        case "h": return "えいち"
        case "i": return "あい"
        case "j": return "じぇー"
        case "k": return "けー"
        case "l": return "える"
        case "m": return "えむ"
        case "n": return "えぬ"
        case "o": return "おー"
        case "p": return "ぴー"
        case "q": return "きゅー"
        case "r": return "あーる"
        case "s": return "えす"
        case "t": return "てぃー"
        case "u": return "ゆー"
        case "v": return "ぶい"
        case "w": return "だぶりゅー"
        case "x": return "えっくす"
        case "y": return "わい"
        case "z": return "ぜっと"
        default: return nil
        }
    }

    // MARK: - 綴りの規則による読み

    /// 子音（カナの行）と母音（a i u e o、長音は "-"、子音の後の「ゆー」は "Y"）の列
    enum Unit: Equatable {
        case consonant(String)
        case vowel(String)
        case sokuon
        case moraicN
    }

    /// 綴りの規則で作る読み（ひらがな）。曖昧な規則（ow → オー／アウ など）は分岐させ、先頭が既定の読み。最大 8 通り
    public static func phonics(_ word: String) -> [String] {
        let letters = Array(word.lowercased()).filter { $0.isLetter && $0.isASCII }
        if letters.isEmpty {
            return []
        }
        var results: [[Unit]] = []
        parse(letters, 0, [], &results)
        var out: [String] = []
        for units in results {
            let kana = ViterbiMorphology.katakanaToHiragana(assemble(units))
            if kana.isEmpty != true && out.contains(kana) != true {
                out.append(kana)
            }
        }
        return out
    }

    static let maxVariants = 8

    static func isVowelLetter(_ c: Character) -> Bool {
        return "aeiou".contains(c)
    }

    /// i 番目から先を読み、分岐ごとに results へ追加する（先に追加されたものほど既定の読み）
    static func parse(_ w: [Character], _ i: Int, _ acc: [Unit], _ results: inout [[Unit]]) {
        if maxVariants <= results.count {
            return
        }
        if w.count <= i {
            results.append(acc)
            return
        }
        func at(_ k: Int) -> Character? {
            return (0 <= k && k < w.count) ? w[k] : nil
        }
        func rest(_ s: String) -> Bool {
            let chars = Array(s)
            if w.count < i + chars.count {
                return false
            }
            var k = 0
            while k < chars.count {
                if w[i + k] != chars[k] {
                    return false
                }
                k += 1
            }
            return true
        }
        func go(_ length: Int, _ units: [Unit]) {
            parse(w, i + length, acc + units, &results)
        }
        /// 直前が短母音 1 つ（長音・二重母音・「ゆー」でない）
        func lastIsShortVowel() -> Bool {
            guard case .some(.vowel(let v)) = acc.last, v.count == 1, v != "Y" else {
                return false
            }
            if 2 <= acc.count, case .vowel = acc[acc.count - 2] {
                return false
            }
            return true
        }
        /// 直後が母音字でない（語末または子音字）
        func closedAfter(_ length: Int) -> Bool {
            guard let next = at(i + length) else {
                return true
            }
            return isVowelLetter(next) != true && next != "y"
        }
        let c = w[i]
        let atEnd = i == w.count - 1

        // 語末の e（子音の後）は読まない。語末の le は「ル」
        if c == "e" && atEnd && 2 <= w.count && isVowelLetter(w[i - 1]) != true && acc.isEmpty != true {
            if w[i - 1] == "l" && 3 <= w.count && isVowelLetter(w[i - 2]) != true {
                go(1, [.vowel("u")])
            } else {
                go(1, [])
            }
            return
        }

        if isVowelLetter(c) || (c == "y" && 0 < i) {
            // マジック e: 母音字 + 子音字 1 つ + 語末の e
            if isVowelLetter(c), let c1 = at(i + 1), isVowelLetter(c1) != true, c1 != "y", c1 != "r", at(i + 2) == "e", i + 3 == w.count || at(i + 3) == "s" || at(i + 3) == "d" {
                let longs: [String: [[Unit]]] = [
                    "a": [[.vowel("e"), .vowel("i")], [.vowel("e-")]],
                    "i": [[.vowel("a"), .vowel("i")]],
                    "o": [[.vowel("o-")], [.vowel("o"), .vowel("u")]],
                    "u": [[.vowel("Y-")]],
                    "e": [[.vowel("i-")]],
                ]
                for v in longs[String(c)] ?? [] {
                    go(1, v)
                }
                return
            }
            // r 化母音（直後が母音字でない r）
            if let r = at(i + 1), r == "r", at(i + 2).map({ isVowelLetter($0) || $0 == "y" }) != true {
                switch c {
                case "a": go(2, [.vowel("a-")])
                case "o": go(2, [.vowel("o-")])
                default: go(2, [.vowel("a-")])
                }
                return
            }
            let pairs: [(String, [[Unit]])] = [
                ("igh", [[.vowel("a"), .vowel("i")]]),
                ("ee", [[.vowel("i-")]]),
                ("ea", [[.vowel("i-")], [.vowel("e")]]),
                ("oo", [[.vowel("u-")], [.vowel("u")]]),
                ("ou", [[.vowel("a"), .vowel("u")], [.vowel("u-")]]),
                ("ow", [[.vowel("o-")], [.vowel("a"), .vowel("u")]]),
                ("oa", [[.vowel("o-")]]),
                ("ai", [[.vowel("e"), .vowel("i")], [.vowel("e-")]]),
                ("ay", [[.vowel("e"), .vowel("i")], [.vowel("e-")]]),
                ("au", [[.vowel("o-")]]),
                ("aw", [[.vowel("o-")]]),
                ("ew", [[.vowel("Y-")]]),
                ("ie", [[.vowel("a"), .vowel("i")], [.vowel("i-")]]),
                ("ei", [[.vowel("e"), .vowel("i")], [.vowel("e-")]]),
                ("ey", [[.vowel("e"), .vowel("i")], [.vowel("i-")]]),
                ("oi", [[.vowel("o"), .vowel("i")]]),
                ("oy", [[.vowel("o"), .vowel("i")]]),
                ("ue", [[.vowel("Y-")]]),
            ]
            for (pattern, variants) in pairs where rest(pattern) {
                for v in variants {
                    go(pattern.count, v)
                }
                return
            }
            // 開音節（母音字 + 子音字 1 つ + 母音字）の u は「ユー」（music, human）、o は「オー」も候補にする
            let open = at(i + 1).map { isVowelLetter($0) != true && $0 != "r" && $0 != "y" } == true && at(i + 2).map { isVowelLetter($0) } == true
            switch c {
            case "a": go(1, [.vowel("a")])
            case "e": go(1, [.vowel("e")])
            case "i": go(1, [.vowel("i")])
            case "o":
                go(1, [.vowel("o")])
                if open {
                    go(1, [.vowel("o-")])
                }
            case "u":
                if open {
                    go(1, [.vowel("Y-")])
                }
                go(1, [.vowel("a")])
                go(1, [.vowel("u")])
            default:
                // 母音としての y: 語末で他に母音字が無い短い語（my, by）は「アイ」、それ以外は「イー」/「イ」
                let others = w.contains { isVowelLetter($0) }
                if atEnd {
                    go(1, others ? [.vowel("i-")] : [.vowel("a"), .vowel("i")])
                } else {
                    go(1, [.vowel("i")])
                }
            }
            return
        }

        // 子音字
        // 短母音の直後で、後ろが閉じている破裂音・破擦音は促音を伴う（cat → キャット、back → バック）
        func stop(_ length: Int, _ consonant: String) {
            if lastIsShortVowel() && closedAfter(length) {
                go(length, [.sokuon, .consonant(consonant)])
            } else {
                go(length, [.consonant(consonant)])
            }
        }
        if rest("tch") { go(3, [.sokuon, .consonant("ch")]); return }
        if rest("chr") || rest("chl") { go(2, [.consonant("k")]); return }
        if rest("ch") { stop(2, "ch"); return }
        if rest("sh") { go(2, [.consonant("sh")]); return }
        if rest("ts") && closedAfter(2) { go(2, [.consonant("ts")]); return }
        if rest("th") { go(2, [.consonant("s")]); return }
        if rest("ph") { go(2, [.consonant("f")]); return }
        if rest("ck") { go(2, [.sokuon, .consonant("k")]); return }
        if rest("ng") { go(2, [.moraicN, .consonant("g")]); return }
        if rest("nk") { go(2, [.moraicN, .consonant("k")]); return }
        if rest("qu") { go(2, [.consonant("kw")]); return }
        if rest("wh") { go(2, [.consonant("w")]); return }
        if rest("wr") { go(2, [.consonant("r")]); return }
        if i == 0 && rest("kn") { go(2, [.consonant("n")]); return }
        if rest("gh") {
            if at(i + 2) == nil || at(i + 2) == "t" {
                go(2, [])
            } else {
                go(2, [.consonant("g")])
            }
            return
        }
        // 同じ子音字の重なりは 1 つとして読む（破裂音なら短母音の後で促音）
        if let next = at(i + 1), next == c, isVowelLetter(c) != true {
            switch c {
            case "p", "t", "k", "b", "d", "g":
                go(2, lastIsShortVowel() ? [.sokuon, .consonant(String(c))] : [.consonant(String(c))])
            case "s":
                go(2, lastIsShortVowel() && closedAfter(2) ? [.sokuon, .consonant("s")] : [.consonant("s")])
            default:
                go(2, [.consonant(String(c) == "l" ? "r" : String(c))])
            }
            return
        }
        let softNext = at(i + 1).map { "eiy".contains($0) } ?? false
        switch c {
        case "c":
            if softNext {
                go(1, [.consonant("s")])
            } else {
                stop(1, "k")
            }
        case "g":
            if softNext {
                go(1, [.consonant("j")])
                go(1, [.consonant("g")])
            } else {
                go(1, [.consonant("g")])
            }
        case "x":
            if i == 0 {
                go(1, [.consonant("z")])
            } else {
                go(1, lastIsShortVowel() ? [.sokuon, .consonant("k"), .consonant("s")] : [.consonant("k"), .consonant("s")])
            }
        case "y":
            if rest("you") {
                go(3, [.vowel("Y-")])
            } else {
                go(1, [.consonant("y")])
            }
        case "l", "r":
            go(1, [.consonant("r")])
        case "v":
            go(1, [.consonant("b")])
        case "s":
            // 母音字に挟まれた s は濁ることが多い（music, busy）
            go(1, [.consonant("s")])
            if 0 < i && isVowelLetter(w[i - 1]) && at(i + 1).map({ isVowelLetter($0) }) == true {
                go(1, [.consonant("z")])
            }
        case "p", "t", "k":
            stop(1, String(c))
        case "m":
            if let next = at(i + 1), next == "b" || next == "p" {
                go(1, [.moraicN])
            } else {
                go(1, [.consonant("m")])
            }
        case "n":
            go(1, [.consonant("n")])
        default:
            go(1, [.consonant(String(c))])
        }
    }

    /// 子音行 → 母音 a i u e o のカナ
    static let rows: [String: [String]] = [
        "": ["ア", "イ", "ウ", "エ", "オ"],
        "k": ["カ", "キ", "ク", "ケ", "コ"],
        "g": ["ガ", "ギ", "グ", "ゲ", "ゴ"],
        "s": ["サ", "シ", "ス", "セ", "ソ"],
        "z": ["ザ", "ジ", "ズ", "ゼ", "ゾ"],
        "t": ["タ", "ティ", "トゥ", "テ", "ト"],
        "d": ["ダ", "ディ", "ドゥ", "デ", "ド"],
        "n": ["ナ", "ニ", "ヌ", "ネ", "ノ"],
        "h": ["ハ", "ヒ", "フ", "ヘ", "ホ"],
        "b": ["バ", "ビ", "ブ", "ベ", "ボ"],
        "p": ["パ", "ピ", "プ", "ペ", "ポ"],
        "m": ["マ", "ミ", "ム", "メ", "モ"],
        "y": ["ヤ", "イ", "ユ", "イェ", "ヨ"],
        "r": ["ラ", "リ", "ル", "レ", "ロ"],
        "w": ["ワ", "ウィ", "ウ", "ウェ", "ウォ"],
        "f": ["ファ", "フィ", "フ", "フェ", "フォ"],
        "sh": ["シャ", "シ", "シュ", "シェ", "ショ"],
        "ch": ["チャ", "チ", "チュ", "チェ", "チョ"],
        "j": ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],
        "kw": ["クア", "クイ", "ク", "クエ", "クオ"],
        "ts": ["ツァ", "ツィ", "ツ", "ツェ", "ツォ"],
    ]

    /// 子音の後の「ゆー」（cute → キュート）
    static let palatalU: [String: String] = [
        "k": "キュ", "g": "ギュ", "n": "ニュ", "h": "ヒュ", "b": "ビュ", "p": "ピュ", "m": "ミュ", "r": "リュ",
        "t": "テュ", "d": "デュ", "s": "シュ", "z": "ジュ", "f": "フュ", "kw": "キュ", "y": "ユ", "w": "ユ",
        "sh": "シュ", "ch": "チュ", "j": "ジュ", "": "ユ",
    ]

    /// 母音の付かない子音に添える母音（t, d は「オ」、ch, j は「イ」、それ以外は「ウ」）
    static func epenthetic(_ consonant: String) -> Int {
        switch consonant {
        case "t", "d": return 4
        case "ch", "j": return 1
        default: return 2
        }
    }

    static func assemble(_ units: [Unit]) -> String {
        var out = ""
        var pending: String? = nil
        func flush() {
            if let c = pending {
                if c == "n" {
                    out += "ン"
                } else {
                    out += rows[c]?[epenthetic(c)] ?? ""
                }
                pending = nil
            }
        }
        for unit in units {
            switch unit {
            case .consonant(let c):
                flush()
                pending = c
            case .sokuon:
                flush()
                if out.isEmpty != true && out.hasSuffix("ッ") != true {
                    out += "ッ"
                }
            case .moraicN:
                flush()
                out += "ン"
            case .vowel(let v):
                let long = v.hasSuffix("-")
                let base = long ? String(v.dropLast()) : v
                let c = pending ?? ""
                pending = nil
                if base == "Y" {
                    out += palatalU[c] ?? "ユ"
                } else if let index = "aiueo".firstIndex(of: Character(base)) {
                    out += rows[c]?["aiueo".distance(from: "aiueo".startIndex, to: index)] ?? ""
                }
                if long {
                    out += "ー"
                }
            }
        }
        flush()
        return out
    }
}
