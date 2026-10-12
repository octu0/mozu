import Foundation

/// Hawk 幹の CPU 推論（文全体を一括処理。ストリーミングはしない）
///
/// 線形層は行列積（Gemm）、ゲートと再帰は SIMD8 で処理する。
/// 双方向のとき再帰幅の前半チャネルは順方向、後半は逆方向に走査する（MLX 学習側と同じ割り当て）。
public final class HawkStack: @unchecked Sendable {
    public let weights: HawkStackWeights
    /// 層ごとの c · log σ(Λ)。log a_t = r_t · logDecay
    private let logDecay: [[Float]]

    public init(weights: HawkStackWeights) {
        self.weights = weights
        self.logDecay = weights.layers.map { layer in
            layer.lambda.values.map { HawkConstants.decayExponent * HawkConstants.logSigmoid($0) }
        }
    }

    public var config: HawkStackConfig {
        return weights.config
    }

    /// x: [frames * modelDim]（行優先）→ 同じ形
    public func forward(_ input: [Float], frames: Int) -> [Float] {
        let cfg = weights.config
        let d = cfg.modelDim
        let r = cfg.rnnDim
        let m = cfg.mlpDim
        let hd = cfg.headDim
        precondition(input.count == frames * d, "HawkStack input must be [frames, modelDim]")
        if frames <= 0 {
            return []
        }
        var x = input
        var normed = [Float](repeating: 0.0, count: frames * d)
        var branch = [Float](repeating: 0.0, count: frames * r)
        var z = [Float](repeating: 0.0, count: frames * r)
        var u = [Float](repeating: 0.0, count: frames * r)
        var keepPre = [Float](repeating: 0.0, count: frames * r)
        var admitPre = [Float](repeating: 0.0, count: frames * r)
        var mlpGate = [Float](repeating: 0.0, count: frames * m)
        var mlpUp = [Float](repeating: 0.0, count: frames * m)

        var l = 0
        while l < cfg.numLayers {
            let lw = weights.layers[l]
            Self.rmsNorm(x, lw.norm1.values, &normed, frames: frames, width: d)
            Gemm.multiply(normed, lw.wGate.values, &branch, m: frames, n: r, k: d)
            Self.geluInPlace(&branch)
            Gemm.multiply(normed, lw.wX.values, &z, m: frames, n: r, k: d)
            Self.conv(z, lw.convW.values, lw.convB.values, &u, frames: frames, width: r, left: cfg.convLeft, taps: cfg.convTaps)
            var h = 0
            while h < cfg.heads {
                let offset = h * hd * hd
                u.withUnsafeBufferPointer { ub in
                    lw.wA.values.withUnsafeBufferPointer { wa in
                        keepPre.withUnsafeMutableBufferPointer { out in
                            Gemm.raw(ub.baseAddress! + h * hd, r, wa.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                        }
                    }
                    lw.wI.values.withUnsafeBufferPointer { wi in
                        admitPre.withUnsafeMutableBufferPointer { out in
                            Gemm.raw(ub.baseAddress! + h * hd, r, wi.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                        }
                    }
                }
                h += 1
            }
            scan(layer: l, frames: frames, u: u, keepPre: keepPre, admitPre: admitPre, branch: &branch)
            Gemm.multiply(branch, lw.wO.values, &x, m: frames, n: d, k: r, beta: 1.0)

            Self.rmsNorm(x, lw.norm2.values, &normed, frames: frames, width: d)
            Gemm.multiply(normed, lw.wMLPGate.values, &mlpGate, m: frames, n: m, k: d)
            Gemm.multiply(normed, lw.wMLPUp.values, &mlpUp, m: frames, n: m, k: d)
            Self.geluInPlace(&mlpGate)
            Self.multiplyInPlace(&mlpGate, mlpUp)
            Gemm.multiply(mlpGate, lw.wMLPDown.values, &x, m: frames, n: d, k: m, beta: 1.0)
            l += 1
        }
        return x
    }

    /// RG-LRU: a = exp(r · c · log σ(Λ)), h = a h + √(1 − a²) (i u)。結果を GeLU 枝に掛け込む
    private func scan(layer: Int, frames: Int, u: [Float], keepPre: [Float], admitPre: [Float], branch: inout [Float]) {
        let cfg = weights.config
        let lw = weights.layers[layer]
        let r = cfg.rnnDim
        let half = cfg.bidirectional ? r / 2 : r
        let one = SIMD8<Float>(repeating: 1.0)
        let eps = SIMD8<Float>(repeating: 1e-6)
        logDecay[layer].withUnsafeBufferPointer { decay in
        lw.bA.values.withUnsafeBufferPointer { bA in
        lw.bI.values.withUnsafeBufferPointer { bI in
        u.withUnsafeBufferPointer { ub in
        keepPre.withUnsafeBufferPointer { kp in
        admitPre.withUnsafeBufferPointer { ap in
        branch.withUnsafeMutableBufferPointer { out in
            let dp = decay.baseAddress!
            let bap = bA.baseAddress!
            let bip = bI.baseAddress!
            let up = ub.baseAddress!
            let kpp = kp.baseAddress!
            let app = ap.baseAddress!
            let op = out.baseAddress!
            var c = 0
            while c < r {
                let logA = Self.load(dp + c)
                let keepBias = Self.load(bap + c)
                let admitBias = Self.load(bip + c)
                var state = SIMD8<Float>(repeating: 0.0)
                let backward = (half <= c)
                var step = 0
                while step < frames {
                    let t = backward ? (frames - 1 - step) : step
                    let idx = t * r + c
                    let keep = SIMDMath.sigmoid(Self.load(kpp + idx) + keepBias)
                    let admit = SIMDMath.sigmoid(Self.load(app + idx) + admitBias)
                    let a = SIMDMath.exp(keep * logA)
                    let scale = (one - a * a + eps).squareRoot()
                    state = a * state + scale * (admit * Self.load(up + idx))
                    Self.store(state * Self.load(op + idx), op + idx)
                    step += 1
                }
                c += 8
            }
        }}}}}}}
    }

    // MARK: - Helpers

    static func rmsNorm(_ x: [Float], _ gain: [Float], _ out: inout [Float], frames: Int, width: Int) {
        var f = 0
        while f < frames {
            let base = f * width
            var sumSq: Float = 0.0
            var c = 0
            while c < width {
                sumSq += x[base + c] * x[base + c]
                c += 1
            }
            let scale = 1.0 / sqrtf(sumSq / Float(width) + HawkConstants.rmsNormEpsilon)
            c = 0
            while c < width {
                out[base + c] = x[base + c] * scale * gain[c]
                c += 1
            }
            f += 1
        }
    }

    /// GeLU のシグモイド近似 x σ(1.702 x)（MLX 側と同じ式）
    static func geluInPlace(_ x: inout [Float]) {
        let k = SIMD8<Float>(repeating: 1.702)
        x.withUnsafeMutableBufferPointer { p in
            let base = p.baseAddress!
            var i = 0
            while i + 8 <= p.count {
                let v = load(base + i)
                store(v * SIMDMath.sigmoid(k * v), base + i)
                i += 8
            }
            while i < p.count {
                p[i] = p[i] / (1.0 + expf(-1.702 * p[i]))
                i += 1
            }
        }
    }

    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func load(_ p: UnsafeMutablePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    static func multiplyInPlace(_ x: inout [Float], _ y: [Float]) {
        var i = 0
        while i < x.count {
            x[i] *= y[i]
            i += 1
        }
    }

    /// チャネルごとの時間畳み込み。出力フレーム t は入力フレーム t − left ... t − left + taps − 1 を見る（範囲外は 0）
    static func conv(_ input: [Float], _ convW: [Float], _ convB: [Float], _ out: inout [Float], frames: Int, width: Int, left: Int, taps: Int) {
        input.withUnsafeBufferPointer { ip in
        convW.withUnsafeBufferPointer { wp in
        convB.withUnsafeBufferPointer { cb in
        out.withUnsafeMutableBufferPointer { op in
            let x = ip.baseAddress!
            let w = wp.baseAddress!
            let bias = cb.baseAddress!
            let o = op.baseAddress!
            var t = 0
            while t < frames {
                var c = 0
                while c < width {
                    var acc = load(bias + c)
                    var j = 0
                    while j < taps {
                        let src = t - left + j
                        if 0 <= src && src < frames {
                            acc += load(w + j * width + c) * load(x + src * width + c)
                        }
                        j += 1
                    }
                    store(acc, o + t * width + c)
                    c += 8
                }
                t += 1
            }
        }}}}
    }
}
