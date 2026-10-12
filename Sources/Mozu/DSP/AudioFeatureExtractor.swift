import Foundation

/// 任意サンプリングレートの WAV 音声ファイル解析および 16kHz モノラル変換器
///
/// 外部収録音声（48kHz 16-bit PCM 等）を直接読み込み、
/// 3サンプル平均アンチエイリアシング間引きによって 16kHz モノラル PCM 列を生成する。
public final class WavAudioReader: @unchecked Sendable {

    public init() {}

    /// ファイルパスから WAV データを読み込み 16kHz モノラル浮動小数点列として抽出する
    public func loadWav16k(from path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try parseWav16k(bytes: [UInt8](data))
    }

    /// バイト配列から WAV データをパースし 16kHz モノラル浮動小数点列を生成する
    public func parseWav16k(bytes: [UInt8]) throws -> [Float] {
        // RIFF ヘッダ長（最小 44 バイト）を満たさない破損ファイルを早期除外する
        if bytes.count < 44 {
            throw WavAudioError.invalidFormat("データサイズが不足しています")
        }

        // マジックナンバー "RIFF" および "WAVE" の検証
        if bytes[0] != 0x52 || bytes[1] != 0x49 || bytes[2] != 0x46 || bytes[3] != 0x46 {
            throw WavAudioError.invalidFormat("RIFF マジックナンバーが一致しません")
        }
        if bytes[8] != 0x57 || bytes[9] != 0x41 || bytes[10] != 0x56 || bytes[11] != 0x45 {
            throw WavAudioError.invalidFormat("WAVE 識別子が一致しません")
        }

        var sampleRate: Int = 16000
        var channels: Int = 1
        var bitsPerSample: Int = 16
        var dataOffset: Int = 0
        var dataSize: Int = 0

        var offset = 12
        let limit = bytes.count

        while (offset + 8) <= limit {
            let c0 = bytes[offset]
            let c1 = bytes[offset + 1]
            let c2 = bytes[offset + 2]
            let c3 = bytes[offset + 3]

            let b0 = Int(bytes[offset + 4])
            let b1 = Int(bytes[offset + 5])
            let b2 = Int(bytes[offset + 6])
            let b3 = Int(bytes[offset + 7])
            let chunkSize = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

            // "fmt " チャンク (0x66, 0x6d, 0x74, 0x20)
            if c0 == 0x66 && c1 == 0x6d && c2 == 0x74 && c3 == 0x20 {
                let fmtOffset = offset + 8
                // wBitsPerSample (オフセット 14..15) まで安全に参照できるよう 16 バイト境界を保証する
                if (fmtOffset + 16) <= limit {
                    let ch0 = Int(bytes[fmtOffset + 2])
                    let ch1 = Int(bytes[fmtOffset + 3])
                    channels = ch0 | (ch1 << 8)

                    let sr0 = Int(bytes[fmtOffset + 4])
                    let sr1 = Int(bytes[fmtOffset + 5])
                    let sr2 = Int(bytes[fmtOffset + 6])
                    let sr3 = Int(bytes[fmtOffset + 7])
                    sampleRate = sr0 | (sr1 << 8) | (sr2 << 16) | (sr3 << 24)

                    let bps0 = Int(bytes[fmtOffset + 14])
                    let bps1 = Int(bytes[fmtOffset + 15])
                    bitsPerSample = bps0 | (bps1 << 8)
                }
            }

            // "data" チャンク (0x64, 0x61, 0x74, 0x61)
            if c0 == 0x64 && c1 == 0x61 && c2 == 0x74 && c3 == 0x61 {
                dataOffset = offset + 8
                dataSize = chunkSize
                break
            }

            offset += 8 + chunkSize
            if (chunkSize & 1) != 0 {
                offset += 1
            }
        }

        if dataOffset == 0 {
            throw WavAudioError.invalidFormat("data チャンクが見つかりません")
        }

        // 不正ヘッダーによるゼロ除算や無限ループを防御する
        if sampleRate <= 0 || channels <= 0 {
            throw WavAudioError.invalidFormat("サンプリング周波数またはチャンネル数が無効です")
        }

        let maxAvailable = max(0, limit - dataOffset)
        let actualByteCount = min(dataSize, maxAvailable)

        // 16-bit / 24-bit PCM サンプルの復元
        var rawPCM: [Float] = []
        switch bitsPerSample {
        case 16:
            let bytesPerFrame = 2 * max(1, channels)
            let frameCount = actualByteCount / bytesPerFrame
            rawPCM = [Float](repeating: 0.0, count: frameCount)

            var f = 0
            while f < frameCount {
                let frameByteOffset = dataOffset + (f * bytesPerFrame)
                var mixedSample: Float = 0.0

                var ch = 0
                while ch < channels {
                    let sampleByteOffset = frameByteOffset + (ch * 2)
                    let lo = UInt16(bytes[sampleByteOffset])
                    let hi = UInt16(bytes[sampleByteOffset + 1])
                    let u16Val = lo | (hi << 8)
                    let i16Val = Int16(bitPattern: u16Val)
                    mixedSample += Float(i16Val) / 32768.0
                    ch += 1
                }

                // 複数チャンネルの場合はモノラルに平均合成する
                if 1 < channels {
                    mixedSample /= Float(channels)
                }
                rawPCM[f] = mixedSample
                f += 1
            }
        case 24:
            // 24-bit PCM（高品質スタジオ音声データ）の符号拡張デコード
            let bytesPerFrame = 3 * max(1, channels)
            let frameCount = actualByteCount / bytesPerFrame
            rawPCM = [Float](repeating: 0.0, count: frameCount)

            var f = 0
            while f < frameCount {
                let frameByteOffset = dataOffset + (f * bytesPerFrame)
                var mixedSample: Float = 0.0

                var ch = 0
                while ch < channels {
                    let sampleByteOffset = frameByteOffset + (ch * 3)
                    let b0 = UInt32(bytes[sampleByteOffset])
                    let b1 = UInt32(bytes[sampleByteOffset + 1])
                    let b2 = UInt32(bytes[sampleByteOffset + 2])
                    var u24 = b0 | (b1 << 8) | (b2 << 16)
                    if (u24 & 0x800000) != 0 {
                        u24 |= 0xFF000000
                    }
                    let i32Val = Int32(bitPattern: u24)
                    mixedSample += Float(i32Val) / 8388608.0
                    ch += 1
                }

                if 1 < channels {
                    mixedSample /= Float(channels)
                }
                rawPCM[f] = mixedSample
                f += 1
            }
        default:
            throw WavAudioError.unsupportedBitsPerSample(bitsPerSample)
        }

        // 16kHz へのリサンプリング処理
        return resampleTo16kHz(pcm: rawPCM, sourceSampleRate: sampleRate)
    }

    /// 任意のサンプリングレートから 16kHz モノラル PCM 列を生成する
    public func resampleTo16kHz(pcm: [Float], sourceSampleRate: Int) -> [Float] {
        if sourceSampleRate <= 0 || pcm.isEmpty {
            return []
        }
        if sourceSampleRate == 16000 {
            return pcm
        }

        // 48kHz 音声データでは、
        // 3 サンプル移動平均によるローパスフィルタを適用してエイリアシング歪みを防止する
        if sourceSampleRate == 48000 {
            let outCount = pcm.count / 3
            var out = [Float](repeating: 0.0, count: outCount)
            var m = 0
            while m < outCount {
                let srcIdx = m * 3
                let s0 = pcm[srcIdx]
                var s1 = s0
                if (srcIdx + 1) < pcm.count {
                    s1 = pcm[srcIdx + 1]
                }
                var s2 = s1
                if (srcIdx + 2) < pcm.count {
                    s2 = pcm[srcIdx + 2]
                }
                out[m] = (s0 + s1 + s2) / 3.0
                m += 1
            }
            return out
        }

        // その他のサンプリング周波数に対する線形補間リサンプリング
        let ratio = Float(sourceSampleRate) / 16000.0
        let outCount = Int(Float(pcm.count) / ratio)
        if outCount <= 0 {
            return []
        }

        var out = [Float](repeating: 0.0, count: outCount)
        var m = 0
        while m < outCount {
            let srcPos = Float(m) * ratio
            let i0 = Int(srcPos)
            let i1 = min(pcm.count - 1, i0 + 1)
            let frac = srcPos - Float(i0)
            if i0 < pcm.count {
                out[m] = (1.0 - frac) * pcm[i0] + (frac * pcm[i1])
            }
            m += 1
        }
        return out
    }
}

/// WAV 音声解析エラー
public enum WavAudioError: Error, Equatable {
    case invalidFormat(String)
    case unsupportedBitsPerSample(Int)
}

/// 64 チャンネル対数 Mel スペクトログラム抽出器
///
/// STFT 短時間フーリエ変換および Mel フィルタバンクにより、
/// NeuralVocoder と厳密に同一の周波数ビン配置を持つ音響特徴量を抽出する。
public final class MelSpectrogramExtractor: @unchecked Sendable {

    public let fftSize: Int
    public let fftBins: Int
    public let melChannels: Int
    public let sampleRate: Float
    public let hopSize: Int
    public let frameSize: Int

    private let bitReversedIndices: [Int]
    private let stageTwiddleReal: [Float]
    private let stageTwiddleImag: [Float]
    private let stageTwiddleOffsets: [Int]
    private let window: [Float]
    private let melFilterbankWeights: [Float] // [fftBins * melChannels]

    public init(
        sampleRate: Float = 16000.0,
        melChannels: Int = AudioConfig.melChannels,
        hopSize: Int = AudioConfig.hopSize,
        frameSize: Int = AudioConfig.frameSize,
        fftSize: Int = 512
    ) {
        self.sampleRate = sampleRate
        self.melChannels = melChannels
        self.hopSize = hopSize
        self.frameSize = frameSize
        self.fftSize = fftSize
        self.fftBins = (fftSize / 2) + 1 // 257

        // 1. Cooley-Tukey Radix-2 FFT テーブルの事前計算
        var m = 0
        var temp = fftSize
        while 1 < temp {
            temp = temp >> 1
            m += 1
        }

        var bitRev = [Int](repeating: 0, count: fftSize)
        var i = 0
        while i < fftSize {
            var rev = 0
            var b = 0
            while b < m {
                if (i & (1 << b)) != 0 {
                    rev |= (1 << ((m - 1) - b))
                }
                b += 1
            }
            bitRev[i] = rev
            i += 1
        }
        self.bitReversedIndices = bitRev

        let half = fftSize / 2
        var twReal = [Float](repeating: 0.0, count: half)
        var twImag = [Float](repeating: 0.0, count: half)
        var k = 0
        let factor = (2.0 * Float.pi) / Float(fftSize)
        while k < half {
            let theta = -1.0 * Float(k) * factor
            twReal[k] = cos(theta)
            twImag[k] = sin(theta)
            k += 1
        }

        var stageReal: [Float] = []
        var stageImag: [Float] = []
        var offsets = [Int](repeating: 0, count: m + 1)
        var stage = 1
        while stage <= m {
            offsets[stage] = stageReal.count
            let len = 1 << stage
            let halfLen = len >> 1
            let step = fftSize / len
            var j = 0
            while j < halfLen {
                stageReal.append(twReal[j * step])
                stageImag.append(twImag[j * step])
                j += 1
            }
            stage += 1
        }
        self.stageTwiddleReal = stageReal
        self.stageTwiddleImag = stageImag
        self.stageTwiddleOffsets = offsets

        // 2. Hann 窓の生成 (frameSize 320 点)
        var win = [Float](repeating: 0.0, count: frameSize)
        var n = 0
        while n < frameSize {
            win[n] = 0.5 * (1.0 - cos((2.0 * Float.pi * Float(n)) / Float(frameSize - 1)))
            n += 1
        }
        self.window = win

        // 3. Mel フィルタバンク重み行列の事前計算
        // NeuralVocoder の周波数ビン配置と完全に一致させ、相互変換の整合性を保証する
        let minMel: Float = 0.0
        let maxFreq = sampleRate * 0.5
        let maxMel = 2595.0 * log10(1.0 + (maxFreq / 700.0))

        var melPoints = [Float](repeating: 0.0, count: melChannels + 2)
        let melStep = (maxMel - minMel) / Float(melChannels + 1)
        var mp = 0
        while mp < (melChannels + 2) {
            melPoints[mp] = minMel + (Float(mp) * melStep)
            mp += 1
        }

        var weights = [Float](repeating: 0.0, count: fftBins * melChannels)
        var binIdx = 0
        while binIdx < fftBins {
            let freq = (Float(binIdx) * maxFreq) / Float(fftBins - 1)
            let mel = 2595.0 * log10(1.0 + (freq / 700.0))

            var ch = 0
            while ch < melChannels {
                let left = melPoints[ch]
                let center = melPoints[ch + 1]
                let right = melPoints[ch + 2]

                var w: Float = 0.0
                if left <= mel {
                    if mel <= center {
                        let span = center - left
                        if 1e-6 < span {
                            w = (mel - left) / span
                        }
                    } else {
                        if mel <= right {
                            let span = right - center
                            if 1e-6 < span {
                                w = (right - mel) / span
                            }
                        }
                    }
                }
                weights[(binIdx * melChannels) + ch] = w
                ch += 1
            }
            binIdx += 1
        }
        self.melFilterbankWeights = weights
    }

    /// In-place FFT 計算
    package func computeFFT(real: inout [Float], imag: inout [Float]) {
        var i = 0
        while i < fftSize {
            let rev = bitReversedIndices[i]
            if i < rev {
                let tr = real[i]
                real[i] = real[rev]
                real[rev] = tr

                let ti = imag[i]
                imag[i] = imag[rev]
                imag[rev] = ti
            }
            i += 1
        }

        var stage = 1
        var temp = fftSize
        var numStages = 0
        while 1 < temp {
            temp = temp >> 1
            numStages += 1
        }

        while stage <= numStages {
            let len = 1 << stage
            let halfLen = len >> 1
            let offset = stageTwiddleOffsets[stage]

            var start = 0
            while start < fftSize {
                var j = 0
                while j < halfLen {
                    let uReal = real[start + j]
                    let uImag = imag[start + j]

                    let wr = stageTwiddleReal[offset + j]
                    let wi = stageTwiddleImag[offset + j]

                    let vReal = real[start + j + halfLen]
                    let vImag = imag[start + j + halfLen]

                    let twVReal = (vReal * wr) - (vImag * wi)
                    let twVImag = (vReal * wi) + (vImag * wr)

                    real[start + j] = uReal + twVReal
                    imag[start + j] = uImag + twVImag
                    real[start + j + halfLen] = uReal - twVReal
                    imag[start + j + halfLen] = uImag - twVImag

                    j += 1
                }
                start += len
            }
            stage += 1
        }
    }

    /// 対数メル 1 フレームから線形パワースペクトル（fftBins 本）を近似復元する（フィルタバンクの擬似逆）
    ///
    /// なぜ擬似逆で足りるか: 無声区間の雑音励振では帯域ごとの大まかなスペクトル包絡が再現できれば十分で、
    /// 各 FFT ビンのパワーを、そのビンに掛かるメルフィルタ重みで加重平均したメル帯域パワーで置き換える。
    public func melToLinearPower(logMel: [Float]) -> [Float] {
        var power = [Float](repeating: 0.0, count: fftBins)
        var melPower = [Float](repeating: 0.0, count: melChannels)
        var ch = 0
        while ch < melChannels {
            var v: Float = 0.0
            if ch < logMel.count {
                v = expf(logMel[ch])
            }
            melPower[ch] = v
            ch += 1
        }
        var b = 0
        while b < fftBins {
            var num: Float = 0.0
            var den: Float = 0.0
            ch = 0
            while ch < melChannels {
                let w = melFilterbankWeights[(b * melChannels) + ch]
                if 0.0 < w {
                    num += w * melPower[ch]
                    den += w
                }
                ch += 1
            }
            if 0.0 < den {
                power[b] = num / den
            } else {
                power[b] = 0.0
            }
            b += 1
        }
        return power
    }

    /// 16kHz PCM 音声から 64 チャンネル対数 Mel スペクトログラムを抽出する
    /// Mel フィルタバンク重み [fftBins * melChannels]（MLX 側で同じ Mel を微分可能に計算するために公開）
    public var filterbankWeights: [Float] {
        return melFilterbankWeights
    }

    /// 分析窓（Hann, frameSize 点）
    public var analysisWindow: [Float] {
        return window
    }

    /// 対数 Mel の下限（無音のフロア）
    public static let logMelFloor: Float = 3.8e-4

    public func extractLogMel(pcm: [Float]) -> [[Float]] {
        if pcm.isEmpty {
            return []
        }

        let frameCount = max(1, pcm.count / hopSize)
        var spectrogram = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: frameCount)

        var realBuf = [Float](repeating: 0.0, count: fftSize)
        var imagBuf = [Float](repeating: 0.0, count: fftSize)
        var powerSpectrum = [Float](repeating: 0.0, count: fftBins)

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize

            // 窓関数の乗算とゼロ埋めバッファの構築
            var s = 0
            while s < fftSize {
                if s < frameSize {
                    let sampleIdx = sampleStart + s
                    if sampleIdx < pcm.count {
                        realBuf[s] = pcm[sampleIdx] * window[s]
                    } else {
                        realBuf[s] = 0.0
                    }
                } else {
                    realBuf[s] = 0.0
                }
                imagBuf[s] = 0.0
                s += 1
            }

            // 高速フーリエ変換の実行
            computeFFT(real: &realBuf, imag: &imagBuf)

            // パワースペクトル P[k] = real^2 + imag^2 の計算
            var k = 0
            while k < fftBins {
                let r = realBuf[k]
                let im = imagBuf[k]
                powerSpectrum[k] = (r * r) + (im * im)
                k += 1
            }

            // Mel フィルタバンクの積算および対数化
            var ch = 0
            while ch < melChannels {
                var melEnergy: Float = 0.0
                var b = 0
                while b < fftBins {
                    melEnergy += powerSpectrum[b] * melFilterbankWeights[(b * melChannels) + ch]
                    b += 1
                }

                // なぜ下限値を 3.8e-4 (logf ≈ -7.88) に設定するか:
                // 音響モデルの無音・ノイズフロア（3.8e-4）と厳密に一致させることで、
                // 実録音の無音区間やフォールバック無音において音響抽出の床を揃え、
                // 無音フレームでの過大な負の残差による勾配汚染を完全に根絶するため。
                var clamped = melEnergy
                if clamped.isFinite != true || clamped < 3.8e-4 {
                    clamped = 3.8e-4
                }
                spectrogram[f][ch] = logf(clamped)
                ch += 1
            }

            f += 1
        }

        return spectrogram
    }

    /// 16kHz PCM 音声から各フレームの線形振幅スペクトログラム (STFT magnitude: 257 bins) を抽出する
    public func extractLinearMagnitudeSpectrogram(pcm: [Float]) -> [[Float]] {
        if pcm.isEmpty {
            return []
        }
        let frameCount = max(1, pcm.count / hopSize)
        var spectrogram = [[Float]](repeating: [Float](repeating: 0.0, count: fftBins), count: frameCount)
        var realBuf = [Float](repeating: 0.0, count: fftSize)
        var imagBuf = [Float](repeating: 0.0, count: fftSize)

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize
            var s = 0
            while s < fftSize {
                if s < frameSize {
                    let sampleIdx = sampleStart + s
                    if sampleIdx < pcm.count {
                        realBuf[s] = pcm[sampleIdx] * window[s]
                    } else {
                        realBuf[s] = 0.0
                    }
                } else {
                    realBuf[s] = 0.0
                }
                imagBuf[s] = 0.0
                s += 1
            }

            computeFFT(real: &realBuf, imag: &imagBuf)

            var k = 0
            while k < fftBins {
                let r = realBuf[k]
                let im = imagBuf[k]
                spectrogram[f][k] = sqrtf(r * r + im * im)
                k += 1
            }
            f += 1
        }
        return spectrogram
    }

    /// 16kHz PCM 音声から各フレームのスペクトル重心（Spectral Centroid: Hz）列を算出する
    public func computeSpectralCentroids(pcm: [Float]) -> [Float] {
        if pcm.isEmpty {
            return []
        }
        let frameCount = max(1, pcm.count / hopSize)
        var centroids = [Float](repeating: 0.0, count: frameCount)
        var realBuf = [Float](repeating: 0.0, count: fftSize)
        var imagBuf = [Float](repeating: 0.0, count: fftSize)

        let binFreqFactor = sampleRate / Float(fftSize)

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize
            var s = 0
            while s < fftSize {
                let pcmIdx = sampleStart + s
                if pcmIdx < pcm.count && s < frameSize {
                    realBuf[s] = pcm[pcmIdx] * window[s]
                } else {
                    realBuf[s] = 0.0
                }
                imagBuf[s] = 0.0
                s += 1
            }

            computeFFT(real: &realBuf, imag: &imagBuf)

            var num: Float = 0.0
            var den: Float = 0.0
            var b = 0
            while b < fftBins {
                let r = realBuf[b]
                let im = imagBuf[b]
                let mag = sqrtf(r * r + im * im)
                let freq = Float(b) * binFreqFactor
                num += freq * mag
                den += mag
                b += 1
            }

            if 1e-6 < den {
                centroids[f] = num / den
            }

            f += 1
        }
        return centroids
    }

    /// 16kHz PCM 音声区間からスペクトル重心（Spectral Centroid: Hz）を算出する
    public func computeSpectralCentroid(pcm: [Float]) -> Float {
        if pcm.isEmpty {
            return 0.0
        }
        let frameCount = max(1, pcm.count / hopSize)
        var realBuf = [Float](repeating: 0.0, count: fftSize)
        var imagBuf = [Float](repeating: 0.0, count: fftSize)
        var centroidSum: Float = 0.0
        var validFrames: Float = 0.0

        let binFreqFactor = sampleRate / Float(fftSize)

        var f = 0
        while f < frameCount {
            let sampleStart = f * hopSize
            var s = 0
            while s < fftSize {
                let pcmIdx = sampleStart + s
                if pcmIdx < pcm.count && s < frameSize {
                    realBuf[s] = pcm[pcmIdx] * window[s]
                } else {
                    realBuf[s] = 0.0
                }
                imagBuf[s] = 0.0
                s += 1
            }

            computeFFT(real: &realBuf, imag: &imagBuf)

            var num: Float = 0.0
            var den: Float = 0.0
            var b = 0
            while b < fftBins {
                let r = realBuf[b]
                let im = imagBuf[b]
                let mag = sqrtf(r * r + im * im)
                let freq = Float(b) * binFreqFactor
                num += freq * mag
                den += mag
                b += 1
            }

            if 1e-6 < den {
                centroidSum += num / den
                validFrames += 1.0
            }

            f += 1
        }

        if 0.0 < validFrames {
            return centroidSum / validFrames
        }
        return 0.0
    }

    /// PCM 音声区間の平均二乗平方根 (RMS) を算出する
    public static func computeRMS(pcm: [Float]) -> Float {
        if pcm.isEmpty {
            return 0.0
        }
        var sumSq: Float = 0.0
        var i = 0
        while i < pcm.count {
            let v = pcm[i]
            sumSq += v * v
            i += 1
        }
        return sqrtf(sumSq / Float(pcm.count))
    }
}

