#if os(macOS)
import Foundation

/// Mozu WebSocket クライアント
///
/// URLSessionWebSocketTask を用いた Pure Swift による低遅延 WebSocket クライアント。
/// 音声合成リクエストの送信、および WAV/PCM バイナリストリームと完了イベントの受信を管理する。
public final class MozuWebClient: @unchecked Sendable {

    public let url: URL
    private var webSocketTask: URLSessionWebSocketTask?
    private let session: URLSession
    private let queue = DispatchQueue(label: "org.mozu.webclient", qos: .userInteractive)

    public var onStart: ((ServerStartEvent) -> Void)?
    public var onAudioChunk: ((Data) -> Void)?
    public var onDone: ((ServerDoneEvent) -> Void)?
    public var onError: ((String) -> Void)?

    private var isConnected: Bool = false

    /// 初期化
    public init(url: URL) {
        self.url = url
        let configuration = URLSessionConfiguration.default
        self.session = URLSession(configuration: configuration)
    }

    /// 便利イニシャライザ
    public convenience init(host: String = "localhost", port: UInt16 = 8080) {
        let url = URL(string: "ws://\(host):\(port)/ws")!
        self.init(url: url)
    }

    /// 接続を開始する
    public func connect() {
        if isConnected {
            return
        }
        let task = session.webSocketTask(with: url)
        self.webSocketTask = task
        task.resume()
        self.isConnected = true
        listenForMessages()
    }

    /// 接続を切断する
    public func disconnect() {
        if isConnected != true {
            return
        }
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        isConnected = false
    }

    /// 音声合成リクエストを送信する
    public func synthesize(
        text: String,
        speed: Float = 1.0,
        pitch: Float = 1.0,
        mode: String = "stream"
    ) throws {
        let request = SynthesizeRequest(
            type: "synthesize",
            text: text,
            speed: speed,
            pitch: pitch,
            mode: mode
        )
        let data = try JSONEncoder().encode(request)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            return
        }
        let message = URLSessionWebSocketTask.Message.string(jsonString)
        webSocketTask?.send(message) { [weak self] error in
            if let error = error {
                self?.onError?("送信失敗: \(error.localizedDescription)")
            }
        }
    }

    /// 進行中の音声合成ストリーミングの中断・停止を要求する。
    ///
    /// サーバーに対して {"type": "cancel"} を即時送信し、以降の不要な音声フレーム生成・配信を打ち切る。
    public func cancel() throws {
        let request = CancelRequest(type: "cancel")
        let data = try JSONEncoder().encode(request)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            return
        }
        let message = URLSessionWebSocketTask.Message.string(jsonString)
        webSocketTask?.send(message) { [weak self] error in
            if let error = error {
                self?.onError?("キャンセル送信失敗: \(error.localizedDescription)")
            }
        }
    }

    /// メッセージを継続受信する
    private func listenForMessages() {
        guard let task = webSocketTask, isConnected else {
            return
        }

        task.receive { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .failure(let error):
                if self.isConnected {
                    self.onError?("受信エラー: \(error.localizedDescription)")
                }
            case .success(let message):
                switch message {
                case .string(let text):
                    self.handleTextMessage(text)
                case .data(let data):
                    self.onAudioChunk?(data)
                @unknown default:
                    break
                }
                // 次のメッセージを受信
                self.listenForMessages()
            }
        }
    }

    /// テキスト形式の JSON イベントを解析してコールバックを発火
    private func handleTextMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }

        // イベント種別を判定するための予備パース
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            return
        }

        let decoder = JSONDecoder()
        switch type {
        case "start":
            if let event = try? decoder.decode(ServerStartEvent.self, from: data) {
                onStart?(event)
            }
        case "done":
            if let event = try? decoder.decode(ServerDoneEvent.self, from: data) {
                onDone?(event)
            }
        case "error":
            if let event = try? decoder.decode(ServerErrorEvent.self, from: data) {
                onError?(event.message)
            }
        default:
            break
        }
    }
}
#endif
