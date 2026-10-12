import Foundation

/// 重みの密テンソル（行優先）
///
/// JSON へは形状と Float32 リトルエンディアンの base64 として書く。
/// 数値を 10 進文字列で書くより約 4 倍小さく、復号もずっと速い。
public struct Tensor: Sendable, Equatable, Codable {
    public var shape: [Int]
    public var values: [Float]

    public init(shape: [Int], values: [Float]) {
        precondition(shape.reduce(1, *) == values.count, "Tensor shape \(shape) does not match \(values.count) values")
        self.shape = shape
        self.values = values
    }

    public init(zeros shape: [Int]) {
        self.shape = shape
        self.values = [Float](repeating: 0.0, count: shape.reduce(1, *))
    }

    public var count: Int {
        return values.count
    }

    enum CodingKeys: String, CodingKey {
        case shape
        case data
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let shape = try c.decode([Int].self, forKey: .shape)
        let data = try c.decode(Data.self, forKey: .data)
        let count = data.count / MemoryLayout<Float>.size
        var values = [Float](repeating: 0.0, count: count)
        values.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        if shape.reduce(1, *) != count {
            throw DecodingError.dataCorruptedError(forKey: .data, in: c, debugDescription: "Tensor shape \(shape) does not match \(count) values")
        }
        self.shape = shape
        self.values = values
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(shape, forKey: .shape)
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try c.encode(data, forKey: .data)
    }
}
