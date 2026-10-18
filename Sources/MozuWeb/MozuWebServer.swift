import Foundation
#if canImport(Network)
import Network
#endif
import Mozu

/// スレッドセーフな非同期キャンセルトークン
final class CancellationToken: @unchecked Sendable {
    private var _isCancelled: Bool = false
    private let lock = NSLock()

    var isCancelled: Bool {
        lock.lock()
        let val = _isCancelled
        lock.unlock()
        return val
    }

    func cancel() {
        lock.lock()
        _isCancelled = true
        lock.unlock()
    }
}

#if canImport(Network)
/// Mozu WebSocket & HTTP 音声合成サーバー
///
/// 外部依存ゼロの Pure Swift による Network.framework を用いた高速サーバー。
/// 同一ポートでブラウザ向け Web UI の HTTP 配信と、双方向 WebSocket 通信の両方を提供する。
public final class MozuWebServer: @unchecked Sendable {

    public let port: UInt16
    public let host: String
    public let engine: MozuEngine

    private var listener: NWListener?
    private let queue = DispatchQueue(label: "org.mozu.webserver", qos: .userInteractive)
    private let synthQueue = DispatchQueue(label: "org.mozu.webserver.synth", qos: .userInitiated, attributes: .concurrent)
    private let tokensLock = NSLock()
    private var activeTokens: [ObjectIdentifier: CancellationToken] = [:]
    private var isRunning: Bool = false
    private let magicWebSocketGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    /// 初期化
    public init(
        port: UInt16 = 8080,
        host: String = "0.0.0.0",
        engine: MozuEngine = MozuEngine()
    ) {
        self.port = port
        self.host = host
        self.engine = engine
    }

    /// サーバーを開始する
    public func start() throws {
        if isRunning {
            return
        }

        let nwPort = NWEndpoint.Port(rawValue: port)!
        let parameters = NWParameters.tcp
        let nwListener = try NWListener(using: parameters, on: nwPort)

        nwListener.newConnectionHandler = { [weak self] connection in
            self?.handleNewConnection(connection)
        }

        nwListener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                break
            case .failed(let error):
                print("[MozuWebServer] リスナー障害: \(error)")
            case .cancelled:
                break
            default:
                break
            }
        }

        nwListener.start(queue: queue)
        self.listener = nwListener
        self.isRunning = true
    }

    /// サーバーを停止する
    public func stop() {
        if isRunning != true {
            return
        }
        listener?.cancel()
        listener = nil
        tokensLock.lock()
        let tokens = Array(activeTokens.values)
        activeTokens.removeAll()
        tokensLock.unlock()
        for token in tokens {
            token.cancel()
        }
        isRunning = false
    }

    /// 新規接続の受付とルーティング
    private func handleNewConnection(_ connection: NWConnection) {
        connection.start(queue: queue)
        readInitialRequest(connection)
    }

    /// 初回 HTTP リクエストヘッダの読み取り
    private func readInitialRequest(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, context, isComplete, error in
            guard let self = self else { return }

            if error != nil {
                connection.cancel()
                return
            }

            guard let data = data, 0 < data.count,
                  let requestString = String(data: data, encoding: .utf8) else {
                connection.cancel()
                return
            }

            let lines = requestString.components(separatedBy: "\r\n")
            guard 0 < lines.count else {
                connection.cancel()
                return
            }

            let requestLine = lines[0]
            let parts = requestLine.components(separatedBy: " ")
            guard 1 < parts.count else {
                connection.cancel()
                return
            }

            let method = parts[0]
            let path = parts[1]

            // ヘッダーの連想配列化
            var headers: [String: String] = [:]
            var lIdx = 1
            while lIdx < lines.count {
                let line = lines[lIdx]
                if line.isEmpty {
                    break
                }
                let hParts = line.split(separator: ":", maxSplits: 1).map { String($0).trimmingCharacters(in: .whitespaces) }
                if hParts.count == 2 {
                    headers[hParts[0].lowercased()] = hParts[1]
                }
                lIdx += 1
            }

            let upgradeHeader = headers["upgrade"]?.lowercased()
            if upgradeHeader == "websocket" {
                // WebSocket アップグレード要求の処理
                self.handleWebSocketHandshake(connection: connection, headers: headers)
            } else {
                // 通常の HTTP 静的リクエストの処理
                switch method {
                case "GET", "HEAD":
                    self.handleHTTPGet(connection: connection, path: path)
                default:
                    self.sendHTTPResponse(connection: connection, status: "405 Method Not Allowed", contentType: "text/plain", body: Data("Method Not Allowed".utf8))
                }
            }
        }
    }

    /// HTTP GET リクエストの処理（内蔵 Web UI 配信）
    private func handleHTTPGet(connection: NWConnection, path: String) {
        switch path {
        case "/", "/index.html":
            let htmlData = Data(WebClientHTML.content.utf8)
            sendHTTPResponse(connection: connection, status: "200 OK", contentType: "text/html; charset=utf-8", body: htmlData)
        default:
            sendHTTPResponse(connection: connection, status: "404 Not Found", contentType: "text/plain", body: Data("Not Found".utf8))
        }
    }

    /// HTTP レスポンスの送信
    private func sendHTTPResponse(connection: NWConnection, status: String, contentType: String, body: Data) {
        var response = "HTTP/1.1 \(status)\r\n"
        response += "Content-Type: \(contentType)\r\n"
        response += "Content-Length: \(body.count)\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"

        var data = Data(response.utf8)
        data.append(body)

        connection.send(content: data, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    /// WebSocket オープニングハンドシェイク (RFC 6455)
    private func handleWebSocketHandshake(connection: NWConnection, headers: [String: String]) {
        guard let secKey = headers["sec-websocket-key"] else {
            sendHTTPResponse(connection: connection, status: "400 Bad Request", contentType: "text/plain", body: Data("Missing Sec-WebSocket-Key".utf8))
            return
        }

        let combined = secKey + magicWebSocketGUID
        let sha1Digest = PureSHA1.hash(data: Data(combined.utf8))
        let acceptValue = sha1Digest.base64EncodedString()

        var handshake = "HTTP/1.1 101 Switching Protocols\r\n"
        handshake += "Upgrade: websocket\r\n"
        handshake += "Connection: Upgrade\r\n"
        handshake += "Sec-WebSocket-Accept: \(acceptValue)\r\n"
        handshake += "\r\n"

        let handshakeData = Data(handshake.utf8)
        connection.send(content: handshakeData, completion: .contentProcessed { [weak self] error in
            guard let self = self else { return }
            if error != nil {
                connection.cancel()
                return
            }
            // ハンドシェイク完了後、WebSocket フレームの受信待機へ移行
            self.readWebSocketFrames(connection: connection, buffer: Data())
        })
    }

    /// WebSocket フレームの継続読み取り
    private func readWebSocketFrames(connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, context, isComplete, error in
            guard let self = self else { return }

            if error != nil || isComplete {
                self.handleCancel(connection: connection)
                connection.cancel()
                return
            }

            var nextBuffer = buffer
            if let incomingData = data, 0 < incomingData.count {
                nextBuffer.append(incomingData)
            }

            if nextBuffer.isEmpty {
                self.readWebSocketFrames(connection: connection, buffer: nextBuffer)
                return
            }

            // フレーム解析ループ
            while true {
                let parseResult = self.parseWebSocketFrame(data: nextBuffer)
                switch parseResult {
                case .needMoreData:
                    // 次のデータ到着を待機
                    self.readWebSocketFrames(connection: connection, buffer: nextBuffer)
                    return
                case .invalidFrame:
                    connection.cancel()
                    return
                case .frame(let opcode, let payload, let bytesConsumed):
                    nextBuffer.removeSubrange(0..<bytesConsumed)
                    self.handleWebSocketMessage(connection: connection, opcode: opcode, payload: payload)
                }

                if nextBuffer.isEmpty {
                    break
                }
            }

            self.readWebSocketFrames(connection: connection, buffer: nextBuffer)
        }
    }

    private enum FrameParseResult {
        case needMoreData
        case invalidFrame
        case frame(opcode: UInt8, payload: Data, bytesConsumed: Int)
    }

    /// RFC 6455 クライアントフレームの解析とアンマスク
    private func parseWebSocketFrame(data: Data) -> FrameParseResult {
        if data.count < 2 {
            return .needMoreData
        }

        let b0 = data[data.startIndex]
        let b1 = data[data.startIndex + 1]

        let opcode = b0 & 0x0F
        let isMasked = (b1 & 0x80) != 0
        var payloadLen = Int(b1 & 0x7F)

        var headerSize = 2

        switch payloadLen {
        case 126:
            if data.count < 4 {
                return .needMoreData
            }
            let u16 = data.subdata(in: (data.startIndex + 2)..<(data.startIndex + 4)).withUnsafeBytes {
                $0.load(as: UInt16.self).bigEndian
            }
            payloadLen = Int(u16)
            headerSize = 4
        case 127:
            if data.count < 10 {
                return .needMoreData
            }
            let u64 = data.subdata(in: (data.startIndex + 2)..<(data.startIndex + 10)).withUnsafeBytes {
                $0.load(as: UInt64.self).bigEndian
            }
            payloadLen = Int(u64)
            headerSize = 10
        default:
            break
        }

        var maskKey: [UInt8] = []
        if isMasked {
            if data.count < headerSize + 4 {
                return .needMoreData
            }
            maskKey = [
                data[data.startIndex + headerSize],
                data[data.startIndex + headerSize + 1],
                data[data.startIndex + headerSize + 2],
                data[data.startIndex + headerSize + 3]
            ]
            headerSize += 4
        }

        let totalFrameSize = headerSize + payloadLen
        if data.count < totalFrameSize {
            return .needMoreData
        }

        var payload = Data(count: payloadLen)
        if 0 < payloadLen {
            payload.withUnsafeMutableBytes { dstBuf in
                data.withUnsafeBytes { srcBuf in
                    let srcPtr = srcBuf.baseAddress!.advanced(by: headerSize)
                    let dstPtr = dstBuf.baseAddress!
                    dstPtr.copyMemory(from: srcPtr, byteCount: payloadLen)
                }
            }

            if isMasked {
                payload.withUnsafeMutableBytes { pBuf in
                    let ptr = pBuf.bindMemory(to: UInt8.self).baseAddress!
                    var i = 0
                    while i < payloadLen {
                        ptr[i] ^= maskKey[i % 4]
                        i += 1
                    }
                }
            }
        }

        return .frame(opcode: opcode, payload: payload, bytesConsumed: totalFrameSize)
    }

    /// WebSocket メッセージのディスパッチ処理
    private func handleWebSocketMessage(connection: NWConnection, opcode: UInt8, payload: Data) {
        switch opcode {
        case 1: // Text Frame
            guard let text = String(data: payload, encoding: .utf8) else { return }
            handleTextCommand(connection: connection, text: text)
        case 8: // Close Frame
            handleCancel(connection: connection)
            connection.cancel()
        case 9: // Ping Frame -> Pong (opcode 10) を返信
            sendWebSocketFrame(connection: connection, opcode: 10, payload: payload)
        default:
            break
        }
    }

    /// テキスト JSON コマンドの解析とディスパッチ
    private func handleTextCommand(connection: NWConnection, text: String) {
        guard let data = text.data(using: .utf8) else { return }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            sendError(connection: connection, message: "JSON形式が不正です。")
            return
        }

        switch type {
        case "cancel":
            handleCancel(connection: connection)
        case "synthesize":
            handleSynthesize(connection: connection, data: data)
        default:
            sendError(connection: connection, message: "未対応のコマンドタイプです: \(type)")
        }
    }

    /// 接続で進行中の音声合成タスクをキャンセル
    private func handleCancel(connection: NWConnection) {
        let connId = ObjectIdentifier(connection)
        tokensLock.lock()
        let token = activeTokens.removeValue(forKey: connId)
        tokensLock.unlock()
        // ロック解放後に即時キャンセルし、進行中タスクの合成・送出ループを同期的に中断する
        token?.cancel()
    }

    /// 音声合成リクエストの処理
    private func handleSynthesize(connection: NWConnection, data: Data) {
        let request: SynthesizeRequest
        do {
            request = try JSONDecoder().decode(SynthesizeRequest.self, from: data)
        } catch {
            sendError(connection: connection, message: "JSON形式が不正です: \(error.localizedDescription)")
            return
        }

        let synthText = request.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if synthText.isEmpty {
            sendError(connection: connection, message: "合成テキストが空です。")
            return
        }

        let connId = ObjectIdentifier(connection)
        let token = CancellationToken()

        // 以前の進行中タスクがあれば即座にキャンセルし、新規トークンで上書きする
        tokensLock.lock()
        let oldToken = activeTokens[connId]
        activeTokens[connId] = token
        tokensLock.unlock()
        oldToken?.cancel()

        let voiceName = request.voice ?? "female"
        let voiceProfile = (try? engine.voice(named: voiceName)) ?? VoiceProfile.preset(named: voiceName)
        let startTime = PlatformTime.uptimeNanoseconds()

        switch request.mode {
        case "wav":
            synthQueue.async { [weak self, weak connection] in
                guard let self = self, let connection = connection else { return }
                if token.isCancelled {
                    return
                }

                let wavData = self.engine.synthesizeWav(text: synthText, voice: voiceProfile, speed: request.speed, pitch: request.pitch)
                if token.isCancelled {
                    return
                }

                let endTime = PlatformTime.uptimeNanoseconds()
                let elapsedSec = Float(endTime - startTime) / 1_000_000_000.0

                let sampleCount = max(0, (wavData.count - 44) / 2)
                let audioDuration = Float(sampleCount) / self.engine.sampleRate
                var rtf = elapsedSec / max(0.001, audioDuration)
                if rtf.isFinite != true {
                    rtf = 0.0
                }

                let startEvent = ServerStartEvent(type: "start", sampleRate: Int(self.engine.sampleRate), channels: 1, format: "wav")
                self.sendJSON(connection: connection, event: startEvent)

                if token.isCancelled {
                    return
                }

                // WAV バイナリフレーム (opcode 2) の送信
                self.sendWebSocketFrame(connection: connection, opcode: 2, payload: wavData)

                if token.isCancelled {
                    return
                }

                let doneEvent = ServerDoneEvent(type: "done", duration: audioDuration, samples: sampleCount, rtf: rtf)
                self.sendJSON(connection: connection, event: doneEvent)

                self.tokensLock.lock()
                if self.activeTokens[connId] === token {
                    self.activeTokens.removeValue(forKey: connId)
                }
                self.tokensLock.unlock()
            }

        default:
            // ストリーミング PCM 逐次配信モード
            let startEvent = ServerStartEvent(type: "start", sampleRate: Int(self.engine.sampleRate), channels: 1, format: "pcm_f32")
            sendJSON(connection: connection, event: startEvent)

            synthQueue.async { [weak self, weak connection] in
                guard let self = self, let connection = connection else { return }

                var totalSamples = 0
                let samples = self.engine.synthesizeStream(
                    text: synthText,
                    voice: voiceProfile,
                    speed: request.speed,
                    pitch: request.pitch,
                    isCancelled: {
                        token.isCancelled
                    },
                    onFrame: { [weak self, weak connection] chunk in
                        guard let self = self, let connection = connection else { return }
                        // キャンセル済みの場合は以降のフレーム送出を打ち切る
                        if token.isCancelled {
                            return
                        }
                        let byteCount = chunk.count * MemoryLayout<Float>.size
                        let chunkData = chunk.withUnsafeBufferPointer { buf in
                            Data(bytes: buf.baseAddress!, count: byteCount)
                        }
                        self.sendWebSocketFrame(connection: connection, opcode: 2, payload: chunkData)
                    }
                )

                if token.isCancelled {
                    self.tokensLock.lock()
                    if self.activeTokens[connId] === token {
                        self.activeTokens.removeValue(forKey: connId)
                    }
                    self.tokensLock.unlock()
                    return
                }

                totalSamples = samples
                let endTime = PlatformTime.uptimeNanoseconds()
                let elapsedSec = Float(endTime - startTime) / 1_000_000_000.0
                let audioDuration = Float(totalSamples) / self.engine.sampleRate
                var rtf = elapsedSec / max(0.001, audioDuration)
                if rtf.isFinite != true {
                    rtf = 0.0
                }

                let doneEvent = ServerDoneEvent(type: "done", duration: audioDuration, samples: totalSamples, rtf: rtf)
                self.sendJSON(connection: connection, event: doneEvent)

                self.tokensLock.lock()
                if self.activeTokens[connId] === token {
                    self.activeTokens.removeValue(forKey: connId)
                }
                self.tokensLock.unlock()
            }
        }
    }

    /// エラーイベントの送信
    private func sendError(connection: NWConnection, message: String) {
        let err = ServerErrorEvent(type: "error", message: message)
        sendJSON(connection: connection, event: err)
    }

    /// JSON Text Frame の送信
    private func sendJSON<T: Encodable>(connection: NWConnection, event: T) {
        do {
            let data = try JSONEncoder().encode(event)
            sendWebSocketFrame(connection: connection, opcode: 1, payload: data)
        } catch {
            print("[MozuWebServer] JSONエンコード失敗: \(error)")
        }
    }

    /// サーバーからクライアントへの WebSocket フレーム送信 (RFC 6455 サーバー送信は非マスク)
    public func sendWebSocketFrame(connection: NWConnection, opcode: UInt8, payload: Data) {
        var frame = Data()
        // FIN = 1 (0x80) | opcode
        frame.append(0x80 | (opcode & 0x0F))

        let len = payload.count
        if len < 126 {
            frame.append(UInt8(len))
        } else {
            if len <= 65535 {
                frame.append(126)
                var u16 = UInt16(len).bigEndian
                withUnsafeBytes(of: &u16) { frame.append(contentsOf: $0) }
            } else {
                frame.append(127)
                var u64 = UInt64(len).bigEndian
                withUnsafeBytes(of: &u64) { frame.append(contentsOf: $0) }
            }
        }

        frame.append(payload)
        connection.send(content: frame, completion: .contentProcessed { _ in })
    }
}
#else
/// Mozu WebSocket & HTTP 音声合成サーバー (Linux / POSIX ソケット版)
///
/// なぜ POSIXWebServer に委任するか:
/// Linux / Cloud Run 環境では Apple の Network.framework が利用できないため、
/// POSIX ソケット + DispatchSource 実装（POSIXWebServer）に透過的に移譲し、
/// macOS と同一のパブリック API（init, start, stop, port, host, engine）を提供する。
public final class MozuWebServer: @unchecked Sendable {

    public let port: UInt16
    public let host: String
    public let engine: MozuEngine

    private let posixServer: POSIXWebServer

    public init(
        port: UInt16 = 8080,
        host: String = "0.0.0.0",
        engine: MozuEngine = MozuEngine()
    ) {
        self.port = port
        self.host = host
        self.engine = engine
        self.posixServer = POSIXWebServer(port: port, host: host, engine: engine)
    }

    public func start() throws {
        try posixServer.start()
    }

    public func stop() {
        posixServer.stop()
    }
}
#endif
