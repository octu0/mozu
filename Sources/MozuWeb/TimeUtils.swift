import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

/// プラットフォーム非依存の高精度モノトニック時刻測定
///
/// なぜ分離するか:
/// macOS 専用の clock_gettime_nsec_np(CLOCK_UPTIME_RAW) は Linux 環境（Cloud Run 等）でコンパイル不可となるため、
/// Linux 標準の clock_gettime(CLOCK_MONOTONIC) と条件コンパイルで統合し、RTF（Real-Time Factor）を高精度かつ安全に計測するため。
enum PlatformTime {
    @inline(__always)
    static func uptimeNanoseconds() -> UInt64 {
        #if canImport(Darwin)
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        #else
        var ts = timespec()
        clock_gettime(CLOCK_MONOTONIC, &ts)
        return UInt64(ts.tv_sec) * 1_000_000_000 + UInt64(ts.tv_nsec)
        #endif
    }
}
