import XCTest
@testable import Mozu

final class BatchPlannerTests: XCTestCase {
    func testFrameBudgetLimitsBatchSize() {
        var rng = SystemRandomNumberGenerator()
        let frames = [100, 120, 110, 900, 950, 130, 105, 1000]
        let batches = BatchPlanner.plan(sampleFrames: frames, batchSize: 4, maxFrames: 2000, rng: &rng)
        var seen = Set<Int>()
        for b in batches {
            let longest = b.map { frames[$0] }.max() ?? 0
            XCTAssertLessThanOrEqual(longest * b.count, 2000, "詰め物込みのフレーム数が上限を超えています")
            XCTAssertLessThanOrEqual(b.count, 4)
            for i in b {
                XCTAssertFalse(seen.contains(i))
                seen.insert(i)
            }
        }
        XCTAssertEqual(seen.count, frames.count)
    }
}
