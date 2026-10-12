import XCTest
@testable import Mozu

/// Common 領域（VectorOperations, Types）の網羅的単体テストスイート
final class CommonTests: XCTestCase {

    // MARK: - 1. VectorOperations SIMD8 ベクトル演算テスト

    func testSIMD8VectorOperations() {
        // 8 要素境界（SIMD8 パス）と端数（スカラーフォールバックパス）の両方において
        // 演算結果がスカラー理論値と完全一致することを検証する。
        let testSizes = [0, 1, 7, 8, 15, 16, 31, 32, 33]
        var sIdx = 0
        while sIdx < testSizes.count {
            let count = testSizes[sIdx]
            var a = [Float](repeating: 0.0, count: count)
            var b = [Float](repeating: 0.0, count: count)
            var expectedDot: Float = 0.0
            var expectedSqA: Float = 0.0

            var i = 0
            while i < count {
                let va = (Float(i % 10) * 0.5) - 2.0
                let vb = (Float((i + 3) % 7) * 0.3) - 1.0
                a[i] = va
                b[i] = vb
                expectedDot += va * vb
                expectedSqA += va * va
                i += 1
            }

            if 0 < count {
                let actualDot = a.withUnsafeBufferPointer { pA in
                    b.withUnsafeBufferPointer { pB in
                        VectorOperations.dotProduct(a: pA.baseAddress!, b: pB.baseAddress!, count: count)
                    }
                }
                let diffDot = abs(actualDot - expectedDot)
                XCTAssertTrue(diffDot < 1e-4, "内積の精度誤差が許容値を超過: size=\(count), diff=\(diffDot)")

                let actualSq = a.withUnsafeBufferPointer { pA in
                    VectorOperations.sumOfSquares(ptr: pA.baseAddress!, count: count)
                }
                let diffSq = abs(actualSq - expectedSqA)
                XCTAssertTrue(diffSq < 1e-4, "二乗和の精度誤差が許容値を超過: size=\(count), diff=\(diffSq)")

                var dstMul = [Float](repeating: 0.0, count: count)
                dstMul.withUnsafeMutableBufferPointer { pDst in
                    a.withUnsafeBufferPointer { pA in
                        b.withUnsafeBufferPointer { pB in
                            VectorOperations.multiply(srcA: pA.baseAddress!, srcB: pB.baseAddress!, dst: pDst.baseAddress!, count: count)
                        }
                    }
                }
                var m = 0
                while m < count {
                    let diffMul = abs(dstMul[m] - (a[m] * b[m]))
                    XCTAssertTrue(diffMul < 1e-5, "要素積の不一致: index=\(m)")
                    m += 1
                }
            }
            sIdx += 1
        }

        // maxMagnitude の検証
        let magData: [Float] = [-1.5, 3.2, -8.7, 4.0, -12.3, 0.0, 7.5, -2.1, 9.4]
        let maxMag = magData.withUnsafeBufferPointer { p in
            VectorOperations.maxMagnitude(ptr: p.baseAddress!, count: magData.count)
        }
        XCTAssertEqual(maxMag, 12.3, accuracy: 1e-5)

        // clamp の検証
        let clampSrc: [Float] = [-2.0, -0.8, -0.3, 0.0, 0.4, 0.7, 1.5]
        var clampDst = [Float](repeating: 0.0, count: clampSrc.count)
        clampDst.withUnsafeMutableBufferPointer { pDst in
            clampSrc.withUnsafeBufferPointer { pSrc in
                VectorOperations.clamp(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: clampSrc.count, minVal: -0.5, maxVal: 0.5)
            }
        }
        let expectedClamp: [Float] = [-0.5, -0.5, -0.3, 0.0, 0.4, 0.5, 0.5]
        var cIdx = 0
        while cIdx < clampSrc.count {
            XCTAssertEqual(clampDst[cIdx], expectedClamp[cIdx], accuracy: 1e-5)
            cIdx += 1
        }

        // softLimitTanh の検証
        let limSrc: [Float] = [-2.0, -0.8, -0.5, 0.0, 0.5, 0.8, 2.0, Float.nan]
        var limDst = [Float](repeating: 0.0, count: limSrc.count)
        limDst.withUnsafeMutableBufferPointer { pDst in
            limSrc.withUnsafeBufferPointer { pSrc in
                VectorOperations.softLimitTanh(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: limSrc.count, threshold: 0.8)
            }
        }
        // 0.8 以下の要素はそのまま保持
        XCTAssertEqual(limDst[1], -0.8, accuracy: 1e-5)
        XCTAssertEqual(limDst[2], -0.5, accuracy: 1e-5)
        XCTAssertEqual(limDst[3], 0.0, accuracy: 1e-5)
        XCTAssertEqual(limDst[4], 0.5, accuracy: 1e-5)
        XCTAssertEqual(limDst[5], 0.8, accuracy: 1e-5)
        // 2.0 の要素は 1.0 未満に滑らかに圧縮
        XCTAssertTrue(limDst[6] < 1.0)
        XCTAssertTrue(0.8 < limDst[6])
        XCTAssertTrue(-1.0 < limDst[0])
        XCTAssertTrue(limDst[0] < -0.8)
        // NaN は 0.0 に安全置換
        XCTAssertEqual(limDst[7], 0.0)

        // quantizeFloatToInt16 の検証
        let quantSrc: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0, Float.nan]
        var quantDst = [Int16](repeating: 0, count: quantSrc.count)
        quantDst.withUnsafeMutableBufferPointer { pDst in
            quantSrc.withUnsafeBufferPointer { pSrc in
                VectorOperations.quantizeFloatToInt16(src: pSrc.baseAddress!, dst: pDst.baseAddress!, count: quantSrc.count)
            }
        }
        XCTAssertEqual(quantDst[0], -32768)
        XCTAssertEqual(quantDst[1], -32767)
        XCTAssertEqual(quantDst[2], 0)
        XCTAssertEqual(quantDst[3], 32767)
        XCTAssertEqual(quantDst[4], 32767)
        XCTAssertEqual(quantDst[5], 0)
    }

    // MARK: - 2. VoiceProfile / AudioConfig 単体検証

    /// 既定の話者プロファイル（女性、男性、中性、子供、重低音）の基音パラメータ整合性を検証
    func testVoiceProfilePresetsAndLabels() throws {
        XCTAssertEqual(VoiceProfile.female.labels["gender"], "female")
        XCTAssertEqual(VoiceProfile.male.labels["gender"], "male")
        XCTAssertTrue(VoiceProfile.neutral.labels.isEmpty)
        let custom = VoiceProfile.preset(named: "gender=female, dialect=標準語")
        XCTAssertEqual(custom.labels["gender"], "female")
        XCTAssertEqual(custom.labels["dialect"], "標準語")
        XCTAssertEqual(VoiceProfile.preset(named: "unknown").labels.count, 0)
        let clamped = VoiceProfile(name: "x", speed: 10.0, pitch: Float.nan)
        XCTAssertEqual(clamped.speed, 2.0)
        XCTAssertEqual(clamped.pitch, 1.0)
        let data = try JSONEncoder().encode(custom)
        let decoded = try JSONDecoder().decode(VoiceProfile.self, from: data)
        XCTAssertEqual(decoded, custom)
    }

    func testAudioConfigConstants() {
        XCTAssertEqual(AudioConfig.sampleRate, 16000)
        XCTAssertEqual(AudioConfig.hopSize, 160)
        XCTAssertEqual(AudioConfig.melChannels, 64)
    }
}
