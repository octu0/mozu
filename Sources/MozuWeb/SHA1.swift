import Foundation

/// Pure Swift による SHA-1 ハッシュ計算器 (FIPS PUB 180-4 準拠)
///
/// なぜ CryptoKit や外部ライブラリを使わないか:
/// Linux 環境（Ubuntu/Debian）や異なる Swift ツールチェーンにおいて CryptoKit のインポート不可や
/// OpenSSL / libcrypto リンクエラーによるビルド破綻を根本から排除し、
/// WebSocket オープニングハンドシェイク（RFC 6455）の Sec-WebSocket-Accept を 100% 確定的に計算するため。
enum PureSHA1 {

    /// 入力データから 20 バイト (160 ビット) の SHA-1 ダイジェストを算出する
    static func hash(data: Data) -> Data {
        var message = [UInt8](data)
        let bitLength = UInt64(message.count) * 8

        // FIPS 180-4 パディング: 先頭に 1 ビット (0x80) を付加
        message.append(0x80)

        // 512 ビット (64 バイト) 境界から 8 バイト手前 (56 mod 64) までゼロ埋めを行う
        while message.count % 64 != 56 {
            message.append(0)
        }

        // メッセージの元ビット長を 64 ビットビッグエンディアンで付加
        var bigBitLength = bitLength.bigEndian
        withUnsafeBytes(of: &bigBitLength) {
            message.append(contentsOf: $0)
        }

        // 初期バッファ定数 (FIPS 180-4 5.1.1)
        var h0: UInt32 = 0x67452301
        var h1: UInt32 = 0xEFCDAB89
        var h2: UInt32 = 0x98BADCFE
        var h3: UInt32 = 0x10325476
        var h4: UInt32 = 0xC3D2E1F0

        let chunkCount = message.count / 64
        var chunkIdx = 0
        while chunkIdx < chunkCount {
            let offset = chunkIdx * 64
            var w = [UInt32](repeating: 0, count: 80)

            // 最初の 16 ワードを 32 ビットビッグエンディアンとして展開
            var t = 0
            while t < 16 {
                let idx = offset + t * 4
                w[t] = (UInt32(message[idx]) << 24)
                     | (UInt32(message[idx + 1]) << 16)
                     | (UInt32(message[idx + 2]) << 8)
                     | UInt32(message[idx + 3])
                t += 1
            }

            // 残り 64 ワードを巡回シフト合成
            while t < 80 {
                let v = w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16]
                w[t] = (v << 1) | (v >> 31)
                t += 1
            }

            var a = h0
            var b = h1
            var c = h2
            var d = h3
            var e = h4

            var i = 0
            while i < 80 {
                let f: UInt32
                let k: UInt32
                switch i {
                case 0...19:
                    f = (b & c) | ((~b) & d)
                    k = 0x5A827999
                case 20...39:
                    f = b ^ c ^ d
                    k = 0x6ED9EBA1
                case 40...59:
                    f = (b & c) | (b & d) | (c & d)
                    k = 0x8F1BBCDC
                default:
                    f = b ^ c ^ d
                    k = 0xCA62C1D6
                }

                let temp = ((a << 5) | (a >> 27)) &+ f &+ e &+ k &+ w[i]
                e = d
                d = c
                c = (b << 30) | (b >> 2)
                b = a
                a = temp
                i += 1
            }

            h0 = h0 &+ a
            h1 = h1 &+ b
            h2 = h2 &+ c
            h3 = h3 &+ d
            h4 = h4 &+ e

            chunkIdx += 1
        }

        var result = Data(capacity: 20)
        var be0 = h0.bigEndian; withUnsafeBytes(of: &be0) { result.append(contentsOf: $0) }
        var be1 = h1.bigEndian; withUnsafeBytes(of: &be1) { result.append(contentsOf: $0) }
        var be2 = h2.bigEndian; withUnsafeBytes(of: &be2) { result.append(contentsOf: $0) }
        var be3 = h3.bigEndian; withUnsafeBytes(of: &be3) { result.append(contentsOf: $0) }
        var be4 = h4.bigEndian; withUnsafeBytes(of: &be4) { result.append(contentsOf: $0) }
        return result
    }
}
