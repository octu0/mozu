import Foundation

/// クライアントからサーバーへの音声合成リクエスト
public struct SynthesizeRequest: Codable, Sendable, Equatable {
    public let type: String
    public let text: String
    public let speed: Float
    public let pitch: Float
    public let mode: String // "stream" または "wav"
    public let voice: String? // "female", "male", "neutral" 等

    public init(
        type: String = "synthesize",
        text: String,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        mode: String = "stream",
        voice: String? = "female"
    ) {
        self.type = type
        self.text = text
        self.speed = speed
        self.pitch = pitch
        self.mode = mode
        self.voice = voice
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case speed
        case pitch
        case mode
        case voice
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decodeIfPresent(String.self, forKey: .type) ?? "synthesize"
        self.text = try container.decode(String.self, forKey: .text)
        self.speed = try container.decodeIfPresent(Float.self, forKey: .speed) ?? 1.0
        self.pitch = try container.decodeIfPresent(Float.self, forKey: .pitch) ?? 1.0
        self.mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "stream"
        self.voice = try container.decodeIfPresent(String.self, forKey: .voice)
    }
}

/// クライアントからサーバーへの合成停止・キャンセルリクエスト
public struct CancelRequest: Codable, Sendable, Equatable {
    public let type: String

    public init(type: String = "cancel") {
        self.type = type
    }
}

/// サーバーからクライアントへの音声ストリーム開始イベント
public struct ServerStartEvent: Codable, Sendable, Equatable {
    public let type: String
    public let sampleRate: Int
    public let channels: Int
    public let format: String

    public init(
        type: String = "start",
        sampleRate: Int = 16000,
        channels: Int = 1,
        format: String = "pcm_f32"
    ) {
        self.type = type
        self.sampleRate = sampleRate
        self.channels = channels
        self.format = format
    }
}

/// サーバーからクライアントへの音声合成完了イベント
public struct ServerDoneEvent: Codable, Sendable, Equatable {
    public let type: String
    public let duration: Float
    public let samples: Int
    public let rtf: Float

    public init(
        type: String = "done",
        duration: Float,
        samples: Int,
        rtf: Float
    ) {
        self.type = type
        self.duration = duration
        self.samples = samples
        self.rtf = rtf
    }
}

/// サーバーからクライアントへのエラー通知イベント
public struct ServerErrorEvent: Codable, Sendable, Equatable {
    public let type: String
    public let message: String

    public init(
        type: String = "error",
        message: String
    ) {
        self.type = type
        self.message = message
    }
}
