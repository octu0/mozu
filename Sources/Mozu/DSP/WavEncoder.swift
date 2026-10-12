import Foundation

/// 16-bit Linear PCM RIFF/WAVE フォーマットエンコーダー
///
/// プラットフォーム固有の音声フレームワークへの依存を排除し、
/// 実行環境を問わずビット完全性を保証した16kHz 16ビットモノラルWAVデータを生成する。
public enum WavEncoder {

    /// 44 バイトの RIFF/WAVE ヘッダを生成
    ///
    /// C言語構造体のパディングによるアライメント不整合を完全に防ぎ、
    /// リトルエンディアン形式で44バイト標準ヘッダを決定論的に構築する。
    public static func createHeader(
        sampleRate: Int = 16000,
        numChannels: Int = 1,
        bitsPerSample: Int = 16,
        dataSize: Int
    ) -> [UInt8] {
        let blockAlign = numChannels * (bitsPerSample / 8)
        let byteRate = sampleRate * blockAlign
        let riffSize = 36 + dataSize

        var header = [UInt8](repeating: 0, count: 44)

        // "RIFF" (0x52, 0x49, 0x46, 0x46)
        header[0] = 0x52
        header[1] = 0x49
        header[2] = 0x46
        header[3] = 0x46

        // ChunkSize (ファイル総サイズ - 8 = 36 + dataSize)
        header[4] = UInt8(riffSize & 0xFF)
        header[5] = UInt8((riffSize >> 8) & 0xFF)
        header[6] = UInt8((riffSize >> 16) & 0xFF)
        header[7] = UInt8((riffSize >> 24) & 0xFF)

        // "WAVE" (0x57, 0x41, 0x56, 0x45)
        header[8] = 0x57
        header[9] = 0x41
        header[10] = 0x56
        header[11] = 0x45

        // "fmt " (0x66, 0x6D, 0x74, 0x20)
        header[12] = 0x66
        header[13] = 0x6D
        header[14] = 0x74
        header[15] = 0x20

        // Subchunk1Size: 16 (リニア PCM のヘッダ長)
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0

        // AudioFormat: 1 (Linear PCM)
        header[20] = 1
        header[21] = 0

        // NumChannels
        header[22] = UInt8(numChannels & 0xFF)
        header[23] = UInt8((numChannels >> 8) & 0xFF)

        // SampleRate (例: 16000 = 0x00003E80)
        header[24] = UInt8(sampleRate & 0xFF)
        header[25] = UInt8((sampleRate >> 8) & 0xFF)
        header[26] = UInt8((sampleRate >> 16) & 0xFF)
        header[27] = UInt8((sampleRate >> 24) & 0xFF)

        // ByteRate (SampleRate * BlockAlign)
        header[28] = UInt8(byteRate & 0xFF)
        header[29] = UInt8((byteRate >> 8) & 0xFF)
        header[30] = UInt8((byteRate >> 16) & 0xFF)
        header[31] = UInt8((byteRate >> 24) & 0xFF)

        // BlockAlign
        header[32] = UInt8(blockAlign & 0xFF)
        header[33] = UInt8((blockAlign >> 8) & 0xFF)

        // BitsPerSample
        header[34] = UInt8(bitsPerSample & 0xFF)
        header[35] = UInt8((bitsPerSample >> 8) & 0xFF)

        // "data" (0x64, 0x61, 0x74, 0x61)
        header[36] = 0x64
        header[37] = 0x61
        header[38] = 0x74
        header[39] = 0x61

        // Subchunk2Size (波形データバイト数)
        header[40] = UInt8(dataSize & 0xFF)
        header[41] = UInt8((dataSize >> 8) & 0xFF)
        header[42] = UInt8((dataSize >> 16) & 0xFF)
        header[43] = UInt8((dataSize >> 24) & 0xFF)

        return header
    }

    /// 浮動小数点 PCM から 16kHz 16-bit Mono WAV データを一括エンコード
    ///
    /// 中間バッファのメモリ確保とコピーを完全にスキップし、
    /// キャッシュミスを最小限に抑えて出力データを即座に構築する。
    public static func encode(
        samples: [Float],
        sampleRate: Int = AudioConfig.sampleRate
    ) -> Data {
        let sampleCount = samples.count
        let bytesPerSample = 2 // 16-bit
        let dataSize = sampleCount * bytesPerSample
        let header = createHeader(sampleRate: sampleRate, numChannels: 1, bitsPerSample: 16, dataSize: dataSize)

        var data = Data(capacity: 44 + dataSize)
        data.append(contentsOf: header)

        if sampleCount <= 0 {
            return data
        }

        data.count += dataSize
        data.withUnsafeMutableBytes { rawBytes in
            let basePtr = rawBytes.baseAddress!
            let pcmDst = basePtr.advanced(by: 44).assumingMemoryBound(to: Int16.self)
            samples.withUnsafeBufferPointer { srcBuf in
                VectorOperations.quantizeFloatToInt16(src: srcBuf.baseAddress!, dst: pcmDst, count: sampleCount)
            }
        }
        return data
    }
}

/// 逐次フレーム書き出しおよびヘッダ更新対応のストリーミング WAV ライター
///
/// 長時間音声の合成において全波形をメモリに保持することなく逐次ディスクへフラッシュし、
/// メモリ消費量を定数領域に抑えながら再生パイプラインへ即時供給する。
public final class WavStreamWriter: @unchecked Sendable {

    private let fileHandle: FileHandle
    public let sampleRate: Int
    private var totalSamplesWritten: Int = 0
    private var isFinalized: Bool = false

    /// 初期化
    ///
    /// ファイル先頭44バイトの領域を確保し、後続のPCMデータを連続オフセットへ追記可能にする。
    public init(fileHandle: FileHandle, sampleRate: Int = AudioConfig.sampleRate) throws {
        self.fileHandle = fileHandle
        self.sampleRate = sampleRate
        let initialHeader = WavEncoder.createHeader(sampleRate: sampleRate, numChannels: 1, bitsPerSample: 16, dataSize: 0)
        try fileHandle.write(contentsOf: initialHeader)
    }

    /// PCM 浮動小数点サンプルチャンクを逐次書き出し
    ///
    /// ボコーダーが生成したフレームバッファを滞留させずに直ちにファイルへ同期する。
    public func write(samples: [Float]) throws {
        if isFinalized {
            return
        }
        let count = samples.count
        if count <= 0 {
            return
        }

        var pcmBuffer = [Int16](repeating: 0, count: count)
        pcmBuffer.withUnsafeMutableBufferPointer { dstBuf in
            samples.withUnsafeBufferPointer { srcBuf in
                VectorOperations.quantizeFloatToInt16(src: srcBuf.baseAddress!, dst: dstBuf.baseAddress!, count: count)
            }
        }

        let byteCount = count * 2
        let chunkData = pcmBuffer.withUnsafeBufferPointer { buf in
            Data(bytes: buf.baseAddress!, count: byteCount)
        }
        try fileHandle.write(contentsOf: chunkData)
        totalSamplesWritten += count
    }

    /// 書き込みを完了し、ファイルヘッダのサイズ情報を確定更新
    ///
    /// 合成完了まで未知である最終波形長をヘッダに正しく記録し、
    /// 音声再生エンジンが再生時間や終端を誤認するのを防ぐ。
    public func finalize() throws {
        if isFinalized {
            return
        }
        isFinalized = true

        let dataSize = totalSamplesWritten * 2
        let finalHeader = WavEncoder.createHeader(
            sampleRate: sampleRate,
            numChannels: 1,
            bitsPerSample: 16,
            dataSize: dataSize
        )

        // ファイル先頭へシークして確定ヘッダを上書き
        try fileHandle.seek(toOffset: 0)
        try fileHandle.write(contentsOf: finalHeader)
        try fileHandle.seekToEnd()
    }
}
