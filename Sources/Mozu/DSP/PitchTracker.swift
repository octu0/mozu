import Foundation

/// ピッチ追跡およびエネルギー抽出の結果
public struct PitchTrackResult: Sendable {
    /// 各フレームの基本周波数 F0 [Hz]（無声区間は 0.0）
    public let f0: [Float]

    /// 各フレームの有声無声フラグ（有声: 1.0, 無声: 0.0）
    public let voiced: [Float]

    /// 各フレームの短時間 RMS 音響エネルギー（0.0 〜 1.0）
    public let energy: [Float]

    /// 総フレーム数
    public let frameCount: Int

    public init(f0: [Float], voiced: [Float], energy: [Float], frameCount: Int) {
        self.f0 = f0
        self.voiced = voiced
        self.energy = energy
        self.frameCount = frameCount
    }
}

/// Pure Swift による正規化交差相関法 (NCCF) 高精度ピッチトラッカー & 音響エネルギー抽出器
///
/// なぜ自己相関法 (NCCF) を採用するか:
/// 1. 外部 C / Python ライブラリ（WORLD, REAPER, PyWorld 等）への依存を完全に排除し、
///    macOS / Linux (Cloud Run) のあらゆる環境で同一の音響韻律特徴量を安定抽出するため。
/// 2. 人間の音声信号において、声帯振動の基本周波数 F0 は波形の周期自己相関の極大点として最も頑健に現れる。
/// 3. 放物線補間（Parabolic Interpolation）を組み合わせることで、
///    16kHz サンプリング時の離散ラグ量子化誤差（数 Hz 〜 10 数 Hz の段差）を解消し、
///    サブサンプル精度の滑らかで生々しい実測 F0 コンターを復元するため。
public final class PitchTracker: @unchecked Sendable {

    /// サンプリング周波数 [Hz]
    public let sampleRate: Float

    /// フレーム長 [サンプル] (25ms = 400 サンプル @ 16kHz)
    public let frameLength: Int

    /// フレームシフト (ホップサイズ) [サンプル] (10ms = 160 サンプル @ 16kHz)
    public let hopSize: Int

    /// 検出最小ピッチ [Hz] (男性の極低音域 60Hz に対応)
    public let minF0: Float

    /// 検出最大ピッチ [Hz] (女性・小児の高音域 500Hz に対応)
    public let maxF0: Float

    /// 探索最小ラグ [サンプル]
    private let minLag: Int

    /// 探索最大ラグ [サンプル]
    private let maxLag: Int

    /// 有声判定の相関閾値
    private let voicingThreshold: Float

    public init(
        sampleRate: Float = 16000.0,
        frameLength: Int = 400,
        hopSize: Int = AudioConfig.hopSize, // 160
        minF0: Float = 60.0,
        maxF0: Float = 500.0,
        voicingThreshold: Float = 0.45
    ) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        self.hopSize = hopSize
        self.minF0 = minF0
        self.maxF0 = maxF0
        self.voicingThreshold = voicingThreshold

        // サンプリング周波数と探索周波数範囲から探索ラグ範囲を確定
        // 60Hz -> 16000 / 60 = 266 サンプル
        // 500Hz -> 16000 / 500 = 32 サンプル
        var lagLow = Int(floor(sampleRate / maxF0))
        if lagLow < 2 {
            lagLow = 2
        }
        self.minLag = lagLow

        var lagHigh = Int(ceil(sampleRate / minF0))
        if (frameLength - 2) < lagHigh {
            lagHigh = frameLength - 2
        }
        self.maxLag = lagHigh
    }

    /// 16kHz PCM 波形全体からフレームごとの実測 F0・有声度・エネルギーを抽出する
    /// 追跡結果を返す。`cleanup` を false にすると跳躍除去・短い有声島の無声化を行わない生の追跡結果を返す（診断用）
    public func track(pcm: [Float], cleanup: Bool = true) -> PitchTrackResult {
        if pcm.isEmpty {
            return PitchTrackResult(f0: [], voiced: [], energy: [], frameCount: 0)
        }

        let frameCount = max(1, pcm.count / hopSize)
        var f0List = [Float](repeating: 0.0, count: frameCount)
        var voicedList = [Float](repeating: 0.0, count: frameCount)
        var energyList = [Float](repeating: 0.0, count: frameCount)

        // フレームごとの分析バッファ
        var frameBuf = [Float](repeating: 0.0, count: frameLength)

        var f = 0
        while f < frameCount {
            let startSample = f * hopSize

            // フレームサンプルの収集
            var s = 0
            while s < frameLength {
                let pcmIdx = startSample + s
                if pcmIdx < pcm.count {
                    frameBuf[s] = pcm[pcmIdx]
                } else {
                    frameBuf[s] = 0.0
                }
                s += 1
            }

            let result = analyzeFrame(samples: frameBuf)
            f0List[f] = result.f0
            voicedList[f] = result.voiced
            energyList[f] = result.energy

            f += 1
        }


        // 時間的連続性に基づく F0 メディアン平滑化 (孤立したピッチ誤検出スパイクの除去)
        var smoothF0 = f0List
        if 2 < frameCount {
            var m = 1
            let mEnd = frameCount - 1
            while m < mEnd {
                // 有声区間内でのみ 3 点メディアンを適用
                let v0 = voicedList[m - 1]
                let v1 = voicedList[m]
                let v2 = voicedList[m + 1]
                if 0.5 <= v0 && 0.5 <= v1 && 0.5 <= v2 {
                    let p0 = f0List[m - 1]
                    let p1 = f0List[m]
                    let p2 = f0List[m + 1]
                    // 3 値の中央値選択
                    var med = p1
                    switch true {
                    case (p0 <= p1 && p1 <= p2) || (p2 <= p1 && p1 <= p0):
                        med = p1
                    case (p1 <= p0 && p0 <= p2) || (p2 <= p0 && p0 <= p1):
                        med = p0
                    default:
                        med = p2
                    }
                    smoothF0[m] = med
                }
                m += 1
            }
        }

        if cleanup != true {
            return PitchTrackResult(f0: smoothF0, voiced: voicedList, energy: energyList, frameCount: frameCount)
        }

        // 有声区間内の局所中央値から大きく外れた F0 跳躍の除去
        let jumpFreeF0 = Self.suppressF0Jumps(f0: smoothF0, voiced: voicedList)
        // 句末のきしみ声などで生じる、文脈と不連続な短い有声島の無声化
        let cleaned = Self.suppressVoicedIslands(f0: jumpFreeF0, voiced: voicedList)

        return PitchTrackResult(
            f0: cleaned.f0,
            voiced: cleaned.voiced,
            energy: energyList,
            frameCount: frameCount
        )
    }

    /// `maxIslandFrames` 以下の短い有声島のうち、前後 `searchFrames` 以内にある長い有声区間（`minRunFrames` 以上）の端の F0 から
    /// `thresholdSemitones` 半音以上離れているもの、または参照できる長い区間が無いものを無声化する
    ///
    /// なぜ必要か:
    /// 読み上げ音声の句末では声がきしみ声に移行し、追跡器が 3〜4 フレームの島として 450 Hz 前後（倍音）や 60〜85 Hz（半周期）を
    /// 断続的に拾う。`suppressF0Jumps` は区間内の跳躍しか扱えず、島全体が外れている場合は残ってしまう。
    /// 島は 30〜40 ms しかなく、周期励振で鳴らすとピッチの飛んだ短い「ピッ」になり、語尾が二重・機械的に聞こえる。
    /// 無声化すれば雑音励振に切り替わり、低エネルギーの Mel と合わせて息の抜けた語尾として自然に収まる。
    /// 文脈と連続する短い島（有声破裂音の前後など）は残す。
    public static func suppressVoicedIslands(
        f0: [Float],
        voiced: [Float],
        maxIslandFrames: Int = 4,
        minRunFrames: Int = 5,
        searchFrames: Int = 30,
        thresholdSemitones: Float = 5.0
    ) -> (f0: [Float], voiced: [Float]) {
        let n = min(f0.count, voiced.count)
        var outF0 = f0
        var outVoiced = voiced
        // 有声区間の列挙
        var runs: [(start: Int, end: Int)] = []
        var start = 0
        while start < n {
            if voiced[start] < 0.5 || f0[start] <= 0.0 {
                start += 1
                continue
            }
            var end = start
            while end < n && 0.5 <= voiced[end] && 0.0 < f0[end] {
                end += 1
            }
            runs.append((start, end))
            start = end
        }
        func edgeMedian(_ run: (start: Int, end: Int), fromEnd: Bool) -> Float {
            let count = min(minRunFrames, run.end - run.start)
            var vals: [Float] = []
            if fromEnd {
                vals = Array(f0[(run.end - count)..<run.end])
            } else {
                vals = Array(f0[run.start..<(run.start + count)])
            }
            vals.sort()
            return vals[vals.count / 2]
        }
        var r = 0
        while r < runs.count {
            let run = runs[r]
            let length = run.end - run.start
            if length <= maxIslandFrames {
                var islandVals = Array(f0[run.start..<run.end])
                islandVals.sort()
                let islandMedian = islandVals[islandVals.count / 2]
                var reference: Float = 0.0
                // 直前の長い区間を優先し、無ければ直後の長い区間を参照する
                var q = r - 1
                while 0 <= q {
                    if minRunFrames <= (runs[q].end - runs[q].start) {
                        if (run.start - runs[q].end) <= searchFrames {
                            reference = edgeMedian(runs[q], fromEnd: true)
                        }
                        break
                    }
                    q -= 1
                }
                if reference <= 0.0 {
                    q = r + 1
                    while q < runs.count {
                        if minRunFrames <= (runs[q].end - runs[q].start) {
                            if (runs[q].start - run.end) <= searchFrames {
                                reference = edgeMedian(runs[q], fromEnd: false)
                            }
                            break
                        }
                        q += 1
                    }
                }
                var drop = false
                if reference <= 0.0 {
                    drop = true
                } else if thresholdSemitones <= abs(12.0 * log2(islandMedian / reference)) {
                    drop = true
                }
                if drop {
                    var t = run.start
                    while t < run.end {
                        outF0[t] = 0.0
                        outVoiced[t] = 0.0
                        t += 1
                    }
                }
            }
            r += 1
        }
        return (outF0, outVoiced)
    }

    /// 有声区間内で、前後 `radius` フレームの局所中央値から `thresholdSemitones` 半音以上離れた F0 を局所中央値で置き換える
    ///
    /// なぜ必要か:
    /// 句末のきしみ声や語頭の立ち上がりで、追跡器が第 2 倍音（約 2 倍）や半周期（約 1/2 倍）を 1〜3 フレーム拾う。
    /// 3 点メディアンは 1 フレームの孤立スパイクしか除けず、読み上げコーパスでは有声区間の約半数が末尾に 5 半音以上の跳躍を残していた。
    /// この跳躍は (1) ボコーダ学習の励振パルス列のピッチ跳び（語尾が二重に聞こえる）、(2) F0 予測器の句末上昇、
    /// (3) アクセント核推定の誤票、の三つにそのまま現れるため、追跡器の出力段で除く。
    /// 置き換え（無声化ではなく）にする理由: きしみ声区間も声道応答は有声のままなので、
    /// 周囲と連続する周期励振を与えるほうが雑音励振に切り替えるより聴感が自然なため。
    public static func suppressF0Jumps(
        f0: [Float],
        voiced: [Float],
        radius: Int = 5,
        thresholdSemitones: Float = 5.0
    ) -> [Float] {
        let n = min(f0.count, voiced.count)
        var out = f0
        var start = 0
        while start < n {
            if voiced[start] < 0.5 || f0[start] <= 0.0 {
                start += 1
                continue
            }
            var end = start
            while end < n && 0.5 <= voiced[end] && 0.0 < f0[end] {
                end += 1
            }
            if 3 <= (end - start) {
                var t = start
                while t < end {
                    // 過去側は置き換え済みの値（out）、未来側は元の値（f0）を参照する。
                    // なぜ: 区間末尾では窓が片側だけになり、誤検出が 3 フレーム続くと元の値だけでは誤検出側が中央値になって
                    // 最後の正しいフレームを誤って置き換えてしまう（391, 443, 443, 86 → 86 が 391 に置き換わる）。
                    var window: [Float] = []
                    var k = max(start, t - radius)
                    while k < min(end, t + radius + 1) {
                        if k < t {
                            window.append(out[k])
                        } else if t < k {
                            window.append(f0[k])
                        }
                        k += 1
                    }
                    if 2 <= window.count {
                        window.sort()
                        let median = window[window.count / 2]
                        if 0.0 < median {
                            let deviation = abs(12.0 * log2(f0[t] / median))
                            if thresholdSemitones <= deviation {
                                out[t] = median
                            }
                        }
                    }
                    t += 1
                }
            }
            start = end
        }
        return out
    }

    /// 単一フレーム（400サンプル）のピッチおよびエネルギー解析
    private func analyzeFrame(samples: [Float]) -> (f0: Float, voiced: Float, energy: Float) {
        // 1. 短時間 RMS エネルギーの計算
        var sumSq: Float = 0.0
        var s = 0
        while s < frameLength {
            let v = samples[s]
            sumSq += v * v
            s += 1
        }
        let rms = sqrtf(sumSq / Float(frameLength))

        // 背景ノイズ・微弱無音区間（約 -46 dBFS 以下）の早期足切り
        // 声帯振動の物理的エネルギーが存在しない無音区間での相関誤検出を防止する
        // なぜ純粋な短時間 RMS（min(1.0, rms)）を返すか:
        // 低レベル DSP 抽出器として人為的な 3 倍増幅を行わず、物理的な信号振幅（0.0〜1.0）を誠実に保持することで、
        // 学習側パイプラインでのピーク正規化や推論側のエネルギー条件付けと数学的にクリーンに連携させるため。
        if rms < 0.005 {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        // 2. 正規化交差相関 (NCCF: Normalized Cross-Correlation) の計算
        // なぜ生サンプル直接の NCCF を用いるか:
        // Hanning 窓を掛けた信号同士の自己相関を取ると、ラグ tau が大きくなるにつれて
        // 窓の重なり形状によるテーパリング減衰（Window Tapering Bias）が生じ、
        // 相関ピークがより小さいラグ（より高い周波数）側へ数 Hz 偏る。
        // 区間長 (frameLength - tau) の生サンプル同士で Pearson 余弦類似度を直接算出することで、
        // 窓関数による周波数バイアスをゼロにし、真の基本周期を正確に特定する。
        let corrSize = maxLag + 2
        var nccf = [Float](repeating: 0.0, count: corrSize)

        var tau = minLag
        while tau <= maxLag {
            let corrLen = frameLength - tau
            var sumCross: Float = 0.0
            var sumEnergy0: Float = 0.0
            var sumEnergyTau: Float = 0.0

            var n = 0
            while n < corrLen {
                let x0 = samples[n]
                let xTau = samples[n + tau]
                sumCross += x0 * xTau
                sumEnergy0 += x0 * x0
                sumEnergyTau += xTau * xTau
                n += 1
            }

            let denom = sqrtf(sumEnergy0 * sumEnergyTau)
            if 1e-8 < denom {
                nccf[tau] = sumCross / denom
            } else {
                nccf[tau] = 0.0
            }

            tau += 1
        }

        // 3. 局所極大ピーク群の検出と大域最大値の特定
        var peakLags: [Int] = []
        var peakVals: [Float] = []
        var globalMaxPeak: Float = -1.0

        var t = minLag + 1
        while t < maxLag {
            let val = nccf[t]
            let prevVal = nccf[t - 1]
            let nextVal = nccf[t + 1]

            // 局所極大値の検出
            if prevVal < val && nextVal <= val {
                peakLags.append(t)
                peakVals.append(val)
                if globalMaxPeak < val {
                    globalMaxPeak = val
                }
            }
            t += 1
        }

        // 有声判定: 大域最大相関が閾値未満の場合は無声音とする
        if globalMaxPeak < voicingThreshold || peakLags.isEmpty {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        // 4. 最短有意ラグ（First Significant Peak）による真の基本周期確定
        // なぜ最短ラグを選択し、フレーム境界相関を排除するか:
        // 基本周波数 F0 は声帯振動信号の「最小周期（最短ラグ）」である。
        // 単純に相関最大値だけを選ぶと、倍音周期（2tau, 3tau）やフレームシフト境界（160サンプル=100Hz）の
        // 構造的アーティファクトに相関ピークが奪われ、オクターブ低周波誤認や 100Hz への縮退を引き起こす。
        // 有声域として十分な強度（voicingThreshold 以上）を持ち、かつ大域最大相関の有意な比率を持つ
        // 最初の（最短ラグの）極大ピークを採用することで、真の声帯振動周期を特定する。
        // 4. 大域相関最大ピークの特定とオクターブ跳躍防止アルゴリズム
        // なぜ単純な最短ラグ優先（First Significant Peak）を排し大域相関最大基準とするか:
        // 短いラグから走査して閾値を超えた最初のピークを選ぶと、真の基本波（相関 0.85〜0.95）が存在するにもかかわらず
        // 手前にある第2倍音フォルマント（相関 0.50〜0.70）を誤認して 400Hz 超へ跳躍するオクターブ倍周波化が発生する。
        // 原則として大域最大相関ピーク（maxLag）を基本波とし、maxLag の約半分（0.44〜0.56倍）に
        // maxVal * 0.92 以上の極めて強力な相関が存在する場合（倍周期誤検出）のみ手前を採用する。
        var maxPeakIdx = -1
        var maxPeakVal: Float = -1.0

        var pIdx = 0
        while pIdx < peakLags.count {
            let pLag = peakLags[pIdx]
            let pVal = peakVals[pIdx]

            let isHopArtifact: Bool
            if (hopSize - 12) <= pLag && pLag <= (hopSize + 12) {
                isHopArtifact = true
            } else {
                isHopArtifact = false
            }

            if isHopArtifact != true {
                if maxPeakVal < pVal {
                    maxPeakVal = pVal
                    maxPeakIdx = pIdx
                }
            }
            pIdx += 1
        }

        var bestLag = 0
        if 0 <= maxPeakIdx {
            let primaryLag = peakLags[maxPeakIdx]
            bestLag = primaryLag

            // 倍周期誤認の検証: primaryLag の約 1/2 または 1/3 に極めて強い先行ピークがあるか検査
            let primaryFloat = Float(primaryLag)
            let subharmonicThreshold = maxPeakVal * 0.92

            var sIdx = 0
            while sIdx < maxPeakIdx {
                let candLag = peakLags[sIdx]
                let candVal = peakVals[sIdx]
                let candFloat = Float(candLag)
                let ratio = candFloat / primaryFloat

                let isHalf: Bool
                if 0.44 <= ratio && ratio <= 0.56 {
                    isHalf = true
                } else {
                    isHalf = false
                }

                let isThird: Bool
                if 0.28 <= ratio && ratio <= 0.38 {
                    isThird = true
                } else {
                    isThird = false
                }

                if (isHalf || isThird) && subharmonicThreshold <= candVal {
                    bestLag = candLag
                    break
                }
                sIdx += 1
            }
        }

        if bestLag <= 0 {
            // ホップ周期近傍しかピークが存在しない場合、真の低域声帯振動（0.65 <= globalMaxPeak）である場合のみ有声採用
            if 0.65 <= globalMaxPeak {
                bestLag = peakLags[0]
            } else {
                return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
            }
        }

        // 6. 放物線補間（Parabolic Interpolation）によるサブサンプル精度ラグ推定
        // 離散ラグ bestLag の前後 3 点から極大放物線の頂点 delta を計算
        let y0 = nccf[bestLag - 1]
        let y1 = nccf[bestLag]
        let y2 = nccf[bestLag + 1]
        let denomParabola = (y0 - (2.0 * y1)) + y2

        var delta: Float = 0.0
        if 1e-6 < abs(denomParabola) {
            delta = (y0 - y2) / (2.0 * denomParabola)
            if delta < -0.5 {
                delta = -0.5
            }
            if 0.5 < delta {
                delta = 0.5
            }
        }

        let preciseLag = Float(bestLag) + delta
        if preciseLag <= 0.0 {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        let extractedF0 = sampleRate / preciseLag

        // 有効ピッチ範囲の最終防壁検査
        if extractedF0 < minF0 || maxF0 < extractedF0 {
            return (f0: 0.0, voiced: 0.0, energy: min(1.0, rms))
        }

        return (
            f0: extractedF0,
            voiced: 1.0,
            energy: min(1.0, rms)
        )
    }
}
