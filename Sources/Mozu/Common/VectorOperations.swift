import Foundation

/// SIMD8 および直接ポインタ操作による高速ベクトル演算
///
/// インスタンス化のオーバーヘッドを完全に排除し、音声波形生成ループや
/// フィルタバンク処理の最深部からインライン展開可能な高効率ユーティリティとして機能させる。
public enum VectorOperations {

    /// 内積（SIMD8 クアッドアキュムレータおよび 32要素タイル展開）
    ///
    /// なぜ 32要素タイルと複数アキュムレータを採用するか:
    /// ニューラルボコーダの隠れ層（256ch 等の 8の倍数）における積和演算で
    /// 命令レベル並列性（ILP）を最大化し、FMA（積和演算）レイテンシを隠蔽するため。
    @inline(__always)
    public static func dotProduct(
        a: UnsafePointer<Float>,
        b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        let rawA = UnsafeRawPointer(a)
        let rawB = UnsafeRawPointer(b)

        var acc0 = SIMD8<Float>(repeating: 0.0)
        var acc1 = SIMD8<Float>(repeating: 0.0)
        var acc2 = SIMD8<Float>(repeating: 0.0)
        var acc3 = SIMD8<Float>(repeating: 0.0)

        let tile32Limit = count - (count % 32)
        var i = 0
        while i < tile32Limit {
            let offset = i * 4
            let va0 = rawA.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
            let vb0 = rawB.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
            acc0 += va0 * vb0

            let va1 = rawA.loadUnaligned(fromByteOffset: offset + 32, as: SIMD8<Float>.self)
            let vb1 = rawB.loadUnaligned(fromByteOffset: offset + 32, as: SIMD8<Float>.self)
            acc1 += va1 * vb1

            let va2 = rawA.loadUnaligned(fromByteOffset: offset + 64, as: SIMD8<Float>.self)
            let vb2 = rawB.loadUnaligned(fromByteOffset: offset + 64, as: SIMD8<Float>.self)
            acc2 += va2 * vb2

            let va3 = rawA.loadUnaligned(fromByteOffset: offset + 96, as: SIMD8<Float>.self)
            let vb3 = rawB.loadUnaligned(fromByteOffset: offset + 96, as: SIMD8<Float>.self)
            acc3 += va3 * vb3

            i += 32
        }

        let tile8Limit = count - (count % 8)
        while i < tile8Limit {
            let offset = i * 4
            let va = rawA.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
            let vb = rawB.loadUnaligned(fromByteOffset: offset, as: SIMD8<Float>.self)
            acc0 += va * vb
            i += 8
        }

        var sum = (acc0 + acc1 + acc2 + acc3).sum()

        while i < count {
            sum += a[i] * b[i]
            i += 1
        }
        return sum
    }

    /// 二乗和
    ///
    /// フレームエネルギー計算および自己相関のラグ0成分導出において、
    /// メモリアクセス回数を半減させキャッシュヒット率を高める。
    @inline(__always)
    public static func sumOfSquares(
        ptr: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        let width = 8
        let limit = count - (count % width)
        var sum: Float = 0.0
        var vecSum = SIMD8<Float>(repeating: 0.0)
        var i = 0

        while i < limit {
            let v = SIMD8<Float>(
                ptr[i + 0], ptr[i + 1], ptr[i + 2], ptr[i + 3],
                ptr[i + 4], ptr[i + 5], ptr[i + 6], ptr[i + 7]
            )
            vecSum += v * v
            i += width
        }
        sum += vecSum.sum()

        while i < count {
            let val = ptr[i]
            sum += val * val
            i += 1
        }
        return sum
    }

    /// 要素ごとの積
    ///
    /// 窓関数の適用やスペクトルゲイン乗算において、
    /// SIMDレジスタから直接メモリへバースト書き込みを行いスループットを最大化する。
    @inline(__always)
    public static func multiply(
        srcA: UnsafePointer<Float>,
        srcB: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        let width = 8
        let limit = count - (count % width)
        var i = 0

        while i < limit {
            let va = SIMD8<Float>(
                srcA[i + 0], srcA[i + 1], srcA[i + 2], srcA[i + 3],
                srcA[i + 4], srcA[i + 5], srcA[i + 6], srcA[i + 7]
            )
            let vb = SIMD8<Float>(
                srcB[i + 0], srcB[i + 1], srcB[i + 2], srcB[i + 3],
                srcB[i + 4], srcB[i + 5], srcB[i + 6], srcB[i + 7]
            )
            let vr = va * vb
            dst[i + 0] = vr[0]
            dst[i + 1] = vr[1]
            dst[i + 2] = vr[2]
            dst[i + 3] = vr[3]
            dst[i + 4] = vr[4]
            dst[i + 5] = vr[5]
            dst[i + 6] = vr[6]
            dst[i + 7] = vr[7]
            i += width
        }

        while i < count {
            dst[i] = srcA[i] * srcB[i]
            i += 1
        }
    }

    /// 絶対値の最大値を検索
    ///
    /// 波形ピーク正規化およびクリッピング検知において、各サンプルの絶対値比較を
    /// 分岐命令なしのSIMD条件置換で行いパイプラインハザードを防ぐ。
    @inline(__always)
    public static func maxMagnitude(
        ptr: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var maxAbs: Float = 0.0
        let width = 8
        let limit = count - (count % width)
        var i = 0
        var maxVec = SIMD8<Float>(repeating: 0.0)
        let zeroVec = SIMD8<Float>(repeating: 0.0)

        while i < limit {
            let v = SIMD8<Float>(
                ptr[i + 0], ptr[i + 1], ptr[i + 2], ptr[i + 3],
                ptr[i + 4], ptr[i + 5], ptr[i + 6], ptr[i + 7]
            )
            let absV = v.replacing(with: -v, where: v .< zeroVec)
            maxVec = maxVec.replacing(with: absV, where: maxVec .< absV)
            i += width
        }

        let m0 = max(maxVec[0], maxVec[1])
        let m1 = max(maxVec[2], maxVec[3])
        let m2 = max(maxVec[4], maxVec[5])
        let m3 = max(maxVec[6], maxVec[7])
        let m01 = max(m0, m1)
        let m23 = max(m2, m3)
        let vecMax = max(m01, m23)
        if maxAbs < vecMax {
            maxAbs = vecMax
        }

        while i < count {
            let absVal = abs(ptr[i])
            if maxAbs < absVal {
                maxAbs = absVal
            }
            i += 1
        }
        return maxAbs
    }

    /// クランプ処理
    ///
    /// 条件分岐の多発によるCPU分岐予測失敗を完全に排除し、
    /// ベクトルレジスタ内で上下限を一括クリッピングする。
    @inline(__always)
    public static func clamp(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int,
        minVal: Float,
        maxVal: Float
    ) {
        let width = 8
        let limit = count - (count % width)
        let minVec = SIMD8<Float>(repeating: minVal)
        let maxVec = SIMD8<Float>(repeating: maxVal)
        var i = 0

        while i < limit {
            let v = SIMD8<Float>(
                src[i + 0], src[i + 1], src[i + 2], src[i + 3],
                src[i + 4], src[i + 5], src[i + 6], src[i + 7]
            )
            var clamped = v.replacing(with: minVec, where: v .< minVec)
            clamped = clamped.replacing(with: maxVec, where: maxVec .< clamped)

            dst[i + 0] = clamped[0]
            dst[i + 1] = clamped[1]
            dst[i + 2] = clamped[2]
            dst[i + 3] = clamped[3]
            dst[i + 4] = clamped[4]
            dst[i + 5] = clamped[5]
            dst[i + 6] = clamped[6]
            dst[i + 7] = clamped[7]
            i += width
        }

        while i < count {
            let v = src[i]
            var c = v
            if v < minVal {
                c = minVal
            }
            if maxVal < c {
                c = maxVal
            }
            dst[i] = c
            i += 1
        }
    }

    /// 双曲線正接関数による平滑サチュレーション制限
    ///
    /// 閾値以下の通常音声サンプルでは線形特性を完全に保持して原音忠実度を担保しつつ、
    /// 閾値を超過した過大振幅のみを滑らかに漸近圧縮して矩形波クリッピング歪みを防止する。
    @inline(__always)
    public static func softLimitTanh(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int,
        threshold: Float = 0.8
    ) {
        let headroom = 1.0 - threshold
        let invHeadroom = 1.0 / headroom
        var i = 0

        while i < count {
            let val = src[i]
            // 不正な浮動小数点例外が後段のPCM量子化やオーディオ出力へ伝播するのを未然に防ぐ。
            if val != val {
                dst[i] = 0.0
                i += 1
                continue
            }

            let absVal = abs(val)
            if absVal <= threshold {
                dst[i] = val
            } else {
                let excess = absVal - threshold
                let compressed = threshold + (headroom * tanh(excess * invHeadroom))
                if val < 0.0 {
                    dst[i] = -compressed
                } else {
                    dst[i] = compressed
                }
            }
            i += 1
        }
    }

    /// 浮動小数点配列を16ビット整数へ変換するSIMD8バルク量子化
    ///
    /// WAV出力直前のPCM整数量子化において、1サンプルごとの変換コストを極小化しつつ、
    /// 予期せぬ例外値に対して無音値を割り当てる安全機構を完結させる。
    @inline(__always)
    public static func quantizeFloatToInt16(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Int16>,
        count: Int
    ) {
        let width = 8
        let limit = count - (count % width)
        let scaleVec = SIMD8<Float>(repeating: 32767.0)
        let minVec = SIMD8<Float>(repeating: -32768.0)
        let maxVec = SIMD8<Float>(repeating: 32767.0)
        var i = 0

        while i < limit {
            let v = SIMD8<Float>(
                src[i + 0], src[i + 1], src[i + 2], src[i + 3],
                src[i + 4], src[i + 5], src[i + 6], src[i + 7]
            )
            var scaled = v * scaleVec

            // IEEE 754浮動小数点の非数特性を利用し、追加の関数呼び出しなしでベクトル並列に非数を検知してゼロに置換する。
            let isNaN = scaled .!= scaled
            scaled = scaled.replacing(with: SIMD8<Float>(repeating: 0.0), where: isNaN)

            scaled = scaled.replacing(with: minVec, where: scaled .< minVec)
            scaled = scaled.replacing(with: maxVec, where: maxVec .< scaled)

            dst[i + 0] = Int16(scaled[0])
            dst[i + 1] = Int16(scaled[1])
            dst[i + 2] = Int16(scaled[2])
            dst[i + 3] = Int16(scaled[3])
            dst[i + 4] = Int16(scaled[4])
            dst[i + 5] = Int16(scaled[5])
            dst[i + 6] = Int16(scaled[6])
            dst[i + 7] = Int16(scaled[7])
            i += width
        }

        while i < count {
            let val = src[i]
            if val != val {
                dst[i] = 0
            } else {
                var s = val * 32767.0
                if s < -32768.0 {
                    s = -32768.0
                }
                if 32767.0 < s {
                    s = 32767.0
                }
                dst[i] = Int16(s)
            }
            i += 1
        }
    }
}
