import Foundation

/// 線形ウォームアップ（lrMin → lrMax）の後、コサイン減衰で lrMin まで下げる。step は 1 始まりのバッチ番号
public struct CosineLRScheduler: Sendable {
    public let lrMax: Float
    public let lrMin: Float

    public init(lrMax: Float, lrMin: Float) {
        self.lrMax = lrMax
        self.lrMin = lrMin
    }

    public func learningRate(step: Int, totalSteps: Int, warmupSteps: Int) -> Float {
        let warmup = max(1, warmupSteps)
        if step <= warmup {
            return lrMin + (lrMax - lrMin) * (Float(step) / Float(warmup))
        }
        let progress = Float(step - warmup) / Float(max(1, totalSteps - warmup))
        let clamped = min(1.0, max(0.0, progress))
        return lrMin + (lrMax - lrMin) * 0.5 * (1.0 + cos(clamped * Float.pi))
    }
}
