import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// 行優先の行列積 c = a · b (+ beta · c)。a [m, k], b [k, n], c [m, n]
///
/// macOS では Accelerate の sgemm、それ以外（Linux）は Pure Swift の SIMD 実装を使う。
public enum Gemm {
    public static func multiply(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    raw(ap.baseAddress!, k, bp.baseAddress!, n, cp.baseAddress!, n, m: m, n: n, k: k, beta: beta)
                }
            }
        }
    }

    public static func raw(_ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int,
                           _ c: UnsafeMutablePointer<Float>, _ ldc: Int, m: Int, n: Int, k: Int, beta: Float) {
        #if canImport(Accelerate)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    1.0, a, Int32(lda), b, Int32(ldb), beta, c, Int32(ldc))
        #else
        var i = 0
        while i < m {
            let cRow = c + i * ldc
            if beta == 0.0 {
                var j = 0
                while j < n {
                    cRow[j] = 0.0
                    j += 1
                }
            } else if beta != 1.0 {
                var j = 0
                while j < n {
                    cRow[j] *= beta
                    j += 1
                }
            }
            var p = 0
            while p < k {
                let aik = a[i * lda + p]
                if aik != 0.0 {
                    let bRow = b + p * ldb
                    let scale = SIMD8<Float>(repeating: aik)
                    var j = 0
                    while j + 8 <= n {
                        let cv = UnsafeRawPointer(cRow + j).loadUnaligned(as: SIMD8<Float>.self)
                        let bv = UnsafeRawPointer(bRow + j).loadUnaligned(as: SIMD8<Float>.self)
                        UnsafeMutableRawPointer(cRow + j).storeBytes(of: cv + scale * bv, as: SIMD8<Float>.self)
                        j += 8
                    }
                    while j < n {
                        cRow[j] += aik * bRow[j]
                        j += 1
                    }
                }
                p += 1
            }
            i += 1
        }
        #endif
    }
}
