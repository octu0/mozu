import XCTest
@testable import Mozu

/// 話し方の潜在変数の分類を検証する
final class StyleClusteringTests: XCTestCase {
    func testKMeansSeparatesClearGroups() {
        var points: [[Float]] = []
        for i in 0..<30 {
            let jitter = Float(i % 5) * 0.01
            points.append([-2 + jitter, 0.0, 1 + jitter])
            points.append([2 + jitter, 0.0, -1 - jitter])
        }
        let (centroids, assignment) = StyleClustering.kMeans(points, k: 2)
        XCTAssertEqual(centroids.count, 2)
        // 交互に並べた 2 群が別々のまとまりになる
        for i in stride(from: 0, to: points.count, by: 2) {
            XCTAssertNotEqual(assignment[i], assignment[i + 1])
            XCTAssertEqual(assignment[i], assignment[0])
        }
        // 動かない次元（2 番目）は代表点でも 0 のまま
        XCTAssertEqual(centroids[0][1], 0, accuracy: 1e-6)
    }

    func testCharacteristicsPreferOverRepresentedLabels() {
        let labels: [[String: String]] = [
            ["tension": "ハイテンション", "speaker": "a"], ["tension": "ハイテンション", "speaker": "a"],
            ["tension": "落ち着いてる", "speaker": "b"], ["tension": "落ち着いてる", "speaker": "b"],
        ]
        let traits = StyleClustering.characteristics(assignment: [0, 0, 1, 1], labels: labels, k: 2)
        XCTAssertEqual(traits[0].first?.hasPrefix("tension=ハイテンション"), true)
        XCTAssertEqual(traits[1].first?.hasPrefix("tension=落ち着いてる"), true)
        XCTAssertFalse(traits.joined().contains { $0.hasPrefix("speaker=") }, "話者は特徴に入れない")
    }
}
