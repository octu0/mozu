import Foundation

/// 東京方言ピッチアクセント則および藤崎モデル風 F0 連続ピッチ輪郭生成器
///
/// SNN音響モデルにおいて声帯振動の基本周波数を滑らかな直流電流注入として
/// ニューロン膜電位に供給することで、機械的な音程段差を排除し自然な抑揚を実現する。
public final class ProsodyModel: Sendable {
    public let baseF0: Float       // 基準ピッチ周波数 (女性声の標準 220.0 Hz)
    public let phraseAmp: Float    // フレーズ成分立ち上がり振幅 (0.18)
    public let phraseDecay: Float  // フレーズ成分減衰率 (2.0 / sec)
    public let accentAmp: Float    // アクセント成分振幅 (0.20)
    public let accentAttack: Float // アクセント立ち上がり係数 (15.0 / sec)
    public let accentDecay: Float  // アクセント立ち下がり減衰係数 (16.0 / sec)

    public init(
        baseF0: Float = 220.0,
        phraseAmp: Float = 0.18,
        phraseDecay: Float = 0.70,
        accentAmp: Float = 0.32,
        accentAttack: Float = 15.0,
        accentDecay: Float = 16.0
    ) {
        self.baseF0 = baseF0
        self.phraseAmp = phraseAmp
        self.phraseDecay = phraseDecay
        self.accentAmp = accentAmp
        self.accentAttack = accentAttack
        self.accentDecay = accentDecay
    }

    /// 東京方言アクセント規則に従って各モーラの High/Low トーンを算出する
    ///
    /// 東京方言の原則に基づき、頭高型、平板型、中高型、尾高型を決定論的に割り振る。
    public func computeMoraTones(moraCount: Int, accentKernel: Int) -> [AccentTone] {
        if moraCount <= 0 {
            return []
        }
        var tones = [AccentTone](repeating: .low, count: moraCount)

        if moraCount == 1 {
            switch accentKernel {
            case 1:
                tones[0] = .high
            default:
                tones[0] = .low
            }
            return tones
        }

        switch accentKernel {
        case 1:
            // 頭高型: 1拍目のみ High, 2拍目以降は Low
            tones[0] = .high
        case 0:
            // 平板型: 1拍目は Low, 2拍目以降はすべて High
            var i = 1
            while i < moraCount {
                tones[i] = .high
                i += 1
            }
        default:
            // 中高型・尾高型: 1拍目は Low, 2拍目から核まで High, 核の後は Low
            var i = 1
            while i < moraCount {
                if i < accentKernel {
                    tones[i] = .high
                }
                i += 1
            }
        }

        return tones
    }

    /// 形態素系列からアクセント句系列を構築する
    ///
    /// 自立語に付属語が連なることで発話句を形成し、
    /// 句境界で呼吸圧成分のリセットを行う。
    public func buildAccentPhrases(morphemes: [Morpheme], vocabulary: PhonemeVocabulary) -> [AccentPhrase] {
        if morphemes.isEmpty {
            return []
        }

        var phrases: [AccentPhrase] = []
        var currentMoras: [MoraToken] = []
        var currentKernel: Int16 = 0

        var mIdx = 0
        while mIdx < morphemes.count {
            let m = morphemes[mIdx]

            // 句読点・記号の場合は現在の句を完了し、ポーズ付きで閉じる
            if m.pos == .symbol {
                if currentMoras.isEmpty != true {
                    // 現在の句にトーンとアクセント核を適用
                    let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
                    var k = 0
                    while k < currentMoras.count {
                        currentMoras[k].tone = tones[k]
                        if 0 < currentKernel && (k + 1) == Int(currentKernel) {
                            currentMoras[k].isAccentKernel = true
                        } else {
                            currentMoras[k].isAccentKernel = false
                        }
                        k += 1
                    }
                    var pauseDur = 30
                    if m.surface == "、" {
                        pauseDur = 15
                    }
                    var isQuestion = false
                    if m.surface == "？" || m.surface == "?" {
                        isQuestion = true
                    }
                    phrases.append(AccentPhrase(moras: currentMoras, pauseAfter: true, pauseDurationFrames: pauseDur, isQuestion: isQuestion))
                    currentMoras = []
                    currentKernel = 0
                }
                mIdx += 1
                continue
            }

            // 新たな自立語（名詞、動詞、形容詞、接続詞、感動詞）で句を分割
            var startsNewPhrase = false
            switch m.pos {
            case .noun, .verb, .adjective, .conjunction, .interjection:
                if currentMoras.isEmpty != true {
                    startsNewPhrase = true
                }
            default:
                break
            }

            if startsNewPhrase {
                let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
                var k = 0
                while k < currentMoras.count {
                    currentMoras[k].tone = tones[k]
                    if 0 < currentKernel && (k + 1) == Int(currentKernel) {
                        currentMoras[k].isAccentKernel = true
                    } else {
                        currentMoras[k].isAccentKernel = false
                    }
                    k += 1
                }
                // 自立語境界では句分割を行いつつ、ポーズ時間は 0 に設定する。
                // 理由: 学習時データにおいて文中の自立語間ポーズは存在せず、
                // ポーズを挿入すると SNN が未知のフレーム・Mel 遷移を出力して発音がブザー化するため。
                phrases.append(AccentPhrase(moras: currentMoras, pauseAfter: false, pauseDurationFrames: 0, isQuestion: false))
                currentMoras = []
                currentKernel = 0
            }

            let wordMoras = vocabulary.kanaToMoras(m.reading)
            if currentMoras.isEmpty {
                currentKernel = m.accentKernel
            }
            currentMoras.append(contentsOf: wordMoras)

            mIdx += 1
        }

        if currentMoras.isEmpty != true {
            let tones = computeMoraTones(moraCount: currentMoras.count, accentKernel: Int(currentKernel))
            var k = 0
            while k < currentMoras.count {
                currentMoras[k].tone = tones[k]
                if 0 < currentKernel && (k + 1) == Int(currentKernel) {
                    currentMoras[k].isAccentKernel = true
                } else {
                    currentMoras[k].isAccentKernel = false
                }
                k += 1
            }
            var isQuestion = false
            if let lastMora = currentMoras.last {
                if lastMora.text == "か" {
                    isQuestion = true
                }
            }
            phrases.append(AccentPhrase(moras: currentMoras, pauseAfter: false, pauseDurationFrames: 0, isQuestion: isQuestion))
        }

        return phrases
    }
}
