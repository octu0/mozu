import Foundation

/// 学習データの話し方の潜在変数 z を分類し、合成で選べる代表点（StyleCluster）を作る
///
/// 各発話の z（参照エンコーダの平均）を求め、k-means でまとめる。まとまりの名前は番号（style1 …）とし、
/// どんな話し方かは、そのまとまりに特に多いラベル（全体での割合に対する倍率が高い値）で示す。
public enum StyleClustering {
    /// 発話ごとの z の平均（読めない発話は nil）
    public static func latents(of utterances: [CorpusUtterance], style: StyleWeights) -> [[Float]?] {
        let encoder = StyleEncoder(weights: style)
        var results = [[Float]?](repeating: nil, count: utterances.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: utterances.count) { i in
            guard let pcm = Corpus.loadPCM(path: utterances[i].wavPath, reader: WavAudioReader()) else {
                return
            }
            let mean = encoder.encode(mel: MelSpectrogramExtractor().extractLogMel(pcm: pcm)).mean
            lock.lock()
            results[i] = mean
            lock.unlock()
        }
        return results
    }

    /// k-means（k-means++ で初期化）。距離は次元ごとに標準偏差で割った空間で測り、
    /// ほとんど動かない次元（標準偏差 minSpread 未満）は使わない
    public static func kMeans(_ points: [[Float]], k: Int, iterations: Int = 50, minSpread: Float = 0.05, seed: UInt64 = 2026) -> (centroids: [[Float]], assignment: [Int]) {
        guard let dim = points.first?.count, k <= points.count, 0 < k else {
            return ([], [])
        }
        var mean = [Float](repeating: 0, count: dim)
        var spread = [Float](repeating: 0, count: dim)
        for p in points {
            for d in 0..<dim {
                mean[d] += p[d]
            }
        }
        for d in 0..<dim {
            mean[d] /= Float(points.count)
        }
        for p in points {
            for d in 0..<dim {
                spread[d] += (p[d] - mean[d]) * (p[d] - mean[d])
            }
        }
        let scale = spread.map { s -> Float in
            let std = sqrt(s / Float(points.count))
            return minSpread <= std ? 1 / std : 0
        }
        let scaled = points.map { p in (0..<dim).map { p[$0] * scale[$0] } }
        func distance(_ a: [Float], _ b: [Float]) -> Float {
            var s: Float = 0
            for d in 0..<dim {
                s += (a[d] - b[d]) * (a[d] - b[d])
            }
            return s
        }
        var rng = SplitMix64(seed: seed)
        var centers = [scaled[Int(rng.next() % UInt64(scaled.count))]]
        var nearest = scaled.map { distance($0, centers[0]) }
        while centers.count < k {
            let total = nearest.reduce(0, +)
            var target = Float(Double(rng.next() % 1_000_000) / 1_000_000.0) * total
            var chosen = scaled.count - 1
            for (i, d) in nearest.enumerated() {
                target -= d
                if target <= 0 {
                    chosen = i
                    break
                }
            }
            centers.append(scaled[chosen])
            for i in 0..<scaled.count {
                nearest[i] = min(nearest[i], distance(scaled[i], centers.last!))
            }
        }
        var assignment = [Int](repeating: 0, count: scaled.count)
        for _ in 0..<iterations {
            var changed = false
            for i in 0..<scaled.count {
                var best = 0
                var bestDistance = Float.greatestFiniteMagnitude
                for c in 0..<k {
                    let d = distance(scaled[i], centers[c])
                    if d < bestDistance {
                        bestDistance = d
                        best = c
                    }
                }
                if assignment[i] != best {
                    assignment[i] = best
                    changed = true
                }
            }
            var sums = [[Float]](repeating: [Float](repeating: 0, count: dim), count: k)
            var counts = [Int](repeating: 0, count: k)
            for i in 0..<scaled.count {
                counts[assignment[i]] += 1
                for d in 0..<dim {
                    sums[assignment[i]][d] += scaled[i][d]
                }
            }
            for c in 0..<k where 0 < counts[c] {
                centers[c] = sums[c].map { $0 / Float(counts[c]) }
            }
            if changed != true {
                break
            }
        }
        // 代表点は元の z の空間で、まとまりに属する発話の平均にする（使わない次元も含める）
        var centroids = [[Float]](repeating: [Float](repeating: 0, count: dim), count: k)
        var counts = [Int](repeating: 0, count: k)
        for i in 0..<points.count {
            counts[assignment[i]] += 1
            for d in 0..<dim {
                centroids[assignment[i]][d] += points[i][d]
            }
        }
        for c in 0..<k where 0 < counts[c] {
            centroids[c] = centroids[c].map { $0 / Float(counts[c]) }
        }
        return (centroids, assignment)
    }

    /// まとまりごとに、全体より特に多いラベル（kind=value、倍率の高い順に最大 limit 件）
    ///
    /// 倍率 = まとまり内の割合 ÷ 全体の割合。まとまりの minShare 以上を占める値だけを候補にする（まれな値の偶然の偏りを除く）
    public static func characteristics(assignment: [Int], labels: [[String: String]], k: Int, excludedKinds: Set<String> = ["speaker"], minShare: Float = 0.2, limit: Int = 4) -> [[String]] {
        var overall: [String: Int] = [:]
        var perCluster = [[String: Int]](repeating: [:], count: k)
        var sizes = [Int](repeating: 0, count: k)
        for (i, item) in labels.enumerated() {
            sizes[assignment[i]] += 1
            for (kind, value) in item where excludedKinds.contains(kind) != true {
                let key = "\(kind)=\(value)"
                overall[key, default: 0] += 1
                perCluster[assignment[i]][key, default: 0] += 1
            }
        }
        let total = Float(max(1, labels.count))
        return (0..<k).map { c in
            let size = Float(max(1, sizes[c]))
            let scored = perCluster[c].compactMap { key, count -> (String, Float)? in
                let share = Float(count) / size
                if share < minShare {
                    return nil
                }
                return (key, share / (Float(overall[key] ?? 1) / total))
            }
            return scored.sorted { $0.1 > $1.1 }.prefix(limit).map { String(format: "%@（×%.1f）", $0.0, $0.1) }
        }
    }
}

/// 再現できる乱数（k-means の初期化用）
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
