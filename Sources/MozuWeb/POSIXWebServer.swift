import Foundation
import Dispatch
import Mozu

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

/// POSIX Berkeley Sockets および libdispatch によるクロスプラットフォーム WebSocket & HTTP 音声合成サーバー
///
/// なぜ POSIX ソケットで実装するか:
/// Cloud Run などの Linux コンテナ環境には macOS 専用の Network.framework (NWListener, NWConnection) が存在しないため、
/// 外部ライブラリ（NIO や Vapor 等）に依存せず、Pure Swift と POSIX 標準システムコール（sys/socket.h）のみで
/// 高速かつ軽量な HTTP / WebSocket 音声合成サーバーを動作させるため。
public final class POSIXWebServer: @unchecked Sendable {

    public let port: UInt16
    public let host: String
    public let engine: MozuEngine

    private let queue = DispatchQueue(label: "org.mozu.posixserver", qos: .userInteractive)
    private let synthQueue = DispatchQueue(label: "org.mozu.posixserver.synth", qos: .userInitiated, attributes: .concurrent)

    private var serverFd: Int32 = -1
    private var listenSource: DispatchSourceRead?
    private var isRunning: Bool = false

    private let stateLock = NSLock()
    private var clientSources: [Int32: DispatchSourceRead] = [:]
    private var clientBuffers: [Int32: Data] = [:]
    private var clientIsWebSocket: [Int32: Bool] = [:]
    private var clientTokens: [Int32: CancellationToken] = [:]

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

    /// サーバーを起動し、指定ポートでリッスンを開始する
    public func start() throws {
        stateLock.lock()
        if isRunning {
            stateLock.unlock()
            return
        }

        #if canImport(Darwin)
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        #else
        let fd = Glibc.socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #endif

        if fd < 0 {
            stateLock.unlock()
            throw NSError(domain: "POSIXWebServer", code: 1, userInfo: [NSLocalizedDescriptionKey: "ソケットの作成に失敗しました (errno: \(errno))"])
        }

        // アドレス再利用を有効化し、再起動時の TIME_WAIT によるバインド失敗を防止
        var opt: Int32 = 1
        #if canImport(Darwin)
        Darwin.setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))
        #else
        Glibc.setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))
        #endif

        // ノンブロッキング化
        #if canImport(Darwin)
        let flags = Darwin.fcntl(fd, F_GETFL, 0)
        _ = Darwin.fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        #else
        let flags = Glibc.fcntl(fd, F_GETFL, 0)
        _ = Glibc.fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        #endif

        var addr = sockaddr_in()
        #if canImport(Darwin)
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        #endif
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian

        if host == "0.0.0.0" {
            addr.sin_addr.s_addr = in_addr_t(0)
        } else {
            #if canImport(Darwin)
            Darwin.inet_pton(AF_INET, host, &addr.sin_addr)
            #else
            Glibc.inet_pton(AF_INET, host, &addr.sin_addr)
            #endif
        }

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                #if canImport(Darwin)
                return Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                #else
                return Glibc.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                #endif
            }
        }

        if bindResult < 0 {
            #if canImport(Darwin)
            Darwin.close(fd)
            #else
            Glibc.close(fd)
            #endif
            stateLock.unlock()
            throw NSError(domain: "POSIXWebServer", code: 2, userInfo: [NSLocalizedDescriptionKey: "ポート \(port) へのバインドに失敗しました (errno: \(errno))"])
        }

        #if canImport(Darwin)
        let listenResult = Darwin.listen(fd, 128)
        #else
        let listenResult = Glibc.listen(fd, 128)
        #endif

        if listenResult < 0 {
            #if canImport(Darwin)
            Darwin.close(fd)
            #else
            Glibc.close(fd)
            #endif
            stateLock.unlock()
            throw NSError(domain: "POSIXWebServer", code: 3, userInfo: [NSLocalizedDescriptionKey: "リッスン開始に失敗しました (errno: \(errno))"])
        }

        self.serverFd = fd
        self.isRunning = true

        let source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        source.setEventHandler { [weak self] in
            self?.acceptConnections()
        }
        source.setCancelHandler {
            #if canImport(Darwin)
            Darwin.close(fd)
            #else
            Glibc.close(fd)
            #endif
        }
        self.listenSource = source
        source.resume()

        stateLock.unlock()
    }

    /// サーバーを停止し、全クライアント接続をクローズする
    public func stop() {
        stateLock.lock()
        if isRunning != true {
            stateLock.unlock()
            return
        }

        listenSource?.cancel()
        listenSource = nil
        serverFd = -1
        isRunning = false

        let fds = Array(clientSources.keys)
        let tokens = Array(clientTokens.values)
        clientSources.removeAll()
        clientBuffers.removeAll()
        clientIsWebSocket.removeAll()
        clientTokens.removeAll()
        stateLock.unlock()

        for token in tokens {
            token.cancel()
        }
        for clientFd in fds {
            #if canImport(Darwin)
            Darwin.close(clientFd)
            #else
            Glibc.close(clientFd)
            #endif
        }
    }

    /// 新規接続の受付ループ
    private func acceptConnections() {
        while true {
            var clientAddr = sockaddr_in()
            var len = socklen_t(MemoryLayout<sockaddr_in>.size)

            let clientFd = withUnsafeMutablePointer(to: &clientAddr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    #if canImport(Darwin)
                    return Darwin.accept(serverFd, $0, &len)
                    #else
                    return Glibc.accept(serverFd, $0, &len)
                    #endif
                }
            }

            if clientFd < 0 {
                // EWOULDBLOCK または EAGAIN で受付可能な接続が尽きたためループ終了
                break
            }

            // ノンブロッキング化
            #if canImport(Darwin)
            let flags = Darwin.fcntl(clientFd, F_GETFL, 0)
            _ = Darwin.fcntl(clientFd, F_SETFL, flags | O_NONBLOCK)
            #else
            let flags = Glibc.fcntl(clientFd, F_GETFL, 0)
            _ = Glibc.fcntl(clientFd, F_SETFL, flags | O_NONBLOCK)
            #endif

            setupClientConnection(clientFd)
        }
    }

    /// クライアント接続の DispatchSource 登録
    private func setupClientConnection(_ clientFd: Int32) {
        let source = DispatchSource.makeReadSource(fileDescriptor: clientFd, queue: queue)

        stateLock.lock()
        clientSources[clientFd] = source
        clientBuffers[clientFd] = Data()
        clientIsWebSocket[clientFd] = false
        stateLock.unlock()

        source.setEventHandler { [weak self] in
            self?.readFromClient(clientFd)
        }

        source.setCancelHandler {
            #if canImport(Darwin)
            Darwin.close(clientFd)
            #else
            Glibc.close(clientFd)
            #endif
        }

        source.resume()
    }

    /// クライアントからの受信データ読み取り
    private func readFromClient(_ clientFd: Int32) {
        var tempBuffer = [UInt8](repeating: 0, count: 8192)
        #if canImport(Darwin)
        let bytesRead = Darwin.recv(clientFd, &tempBuffer, tempBuffer.count, 0)
        #else
        let bytesRead = Glibc.recv(clientFd, &tempBuffer, tempBuffer.count, 0)
        #endif

        if bytesRead <= 0 {
            // 切断またはソケットエラー発生のためリソース解放
            closeClient(clientFd)
            return
        }

        let incomingData = Data(bytes: tempBuffer, count: bytesRead)

        stateLock.lock()
        guard var buf = clientBuffers[clientFd],
              let isWS = clientIsWebSocket[clientFd] else {
            stateLock.unlock()
            return
        }
        buf.append(incomingData)
        clientBuffers[clientFd] = buf
        stateLock.unlock()

        if isWS != true {
            // HTTP リクエスト処理
            handleIncomingHTTP(clientFd: clientFd)
        } else {
            // WebSocket フレーム処理
            handleIncomingWebSocket(clientFd: clientFd)
        }
    }

    /// HTTP リクエストの解析とルーティング
    private func handleIncomingHTTP(clientFd: Int32) {
        stateLock.lock()
        guard let buf = clientBuffers[clientFd] else {
            stateLock.unlock()
            return
        }
        stateLock.unlock()

        guard let requestString = String(data: buf, encoding: .utf8) else {
            closeClient(clientFd)
            return
        }

        // HTTP ヘッダー終端 (\r\n\r\n) を待機
        guard let headerEndRange = requestString.range(of: "\r\n\r\n") else {
            return
        }

        let headerSection = String(requestString[..<headerEndRange.lowerBound])
        let lines = headerSection.components(separatedBy: "\r\n")
        guard 0 < lines.count else {
            closeClient(clientFd)
            return
        }

        let requestLine = lines[0]
        let parts = requestLine.components(separatedBy: " ")
        guard 1 < parts.count else {
            closeClient(clientFd)
            return
        }

        let method = parts[0]
        let path = parts[1]

        var headers: [String: String] = [:]
        var lIdx = 1
        while lIdx < lines.count {
            let line = lines[lIdx]
            let hParts = line.split(separator: ":", maxSplits: 1).map { String($0).trimmingCharacters(in: .whitespaces) }
            if hParts.count == 2 {
                headers[hParts[0].lowercased()] = hParts[1]
            }
            lIdx += 1
        }

        let upgradeHeader = headers["upgrade"]?.lowercased()
        if upgradeHeader == "websocket" {
            // WebSocket ハンドシェイク処理
            upgradeToWebSocket(clientFd: clientFd, headers: headers)
        } else {
            // 通常の HTTP 静的リクエスト配信
            switch method {
            case "GET", "HEAD":
                handleHTTPGet(clientFd: clientFd, path: path)
            default:
                sendHTTPResponse(clientFd: clientFd, status: "405 Method Not Allowed", contentType: "text/plain", body: Data("Method Not Allowed".utf8))
            }
        }
    }

    /// HTTP GET リクエストへの応答
    private func handleHTTPGet(clientFd: Int32, path: String) {
        switch path {
        case "/", "/index.html":
            let htmlData = Data(WebClientHTML.content.utf8)
            sendHTTPResponse(clientFd: clientFd, status: "200 OK", contentType: "text/html; charset=utf-8", body: htmlData)
        default:
            sendHTTPResponse(clientFd: clientFd, status: "404 Not Found", contentType: "text/plain", body: Data("Not Found".utf8))
        }
    }

    /// HTTP レスポンスの送信
    private func sendHTTPResponse(clientFd: Int32, status: String, contentType: String, body: Data) {
        var response = "HTTP/1.1 \(status)\r\n"
        response += "Content-Type: \(contentType)\r\n"
        response += "Content-Length: \(body.count)\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"

        var data = Data(response.utf8)
        data.append(body)

        sendRaw(clientFd: clientFd, data: data)
        closeClient(clientFd)
    }

    /// WebSocket オープニングハンドシェイク
    private func upgradeToWebSocket(clientFd: Int32, headers: [String: String]) {
        guard let secKey = headers["sec-websocket-key"] else {
            sendHTTPResponse(clientFd: clientFd, status: "400 Bad Request", contentType: "text/plain", body: Data("Missing Sec-WebSocket-Key".utf8))
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

        sendRaw(clientFd: clientFd, data: Data(handshake.utf8))

        stateLock.lock()
        clientIsWebSocket[clientFd] = true
        clientBuffers[clientFd] = Data()
        stateLock.unlock()
    }

    /// WebSocket フレームの解析ループ
    private func handleIncomingWebSocket(clientFd: Int32) {
        stateLock.lock()
        guard var buf = clientBuffers[clientFd] else {
            stateLock.unlock()
            return
        }
        stateLock.unlock()

        while true {
            let parseResult = parseWebSocketFrame(data: buf)
            switch parseResult {
            case .needMoreData:
                stateLock.lock()
                clientBuffers[clientFd] = buf
                stateLock.unlock()
                return
            case .invalidFrame:
                closeClient(clientFd)
                return
            case .frame(let opcode, let payload, let bytesConsumed):
                buf.removeSubrange(0..<bytesConsumed)
                dispatchWebSocketMessage(clientFd: clientFd, opcode: opcode, payload: payload)
            }

            if buf.isEmpty {
                break
            }
        }

        stateLock.lock()
        clientBuffers[clientFd] = buf
        stateLock.unlock()
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

    /// WebSocket メッセージのディスパッチ
    private func dispatchWebSocketMessage(clientFd: Int32, opcode: UInt8, payload: Data) {
        switch opcode {
        case 1: // Text Frame
            guard let text = String(data: payload, encoding: .utf8) else { return }
            handleTextCommand(clientFd: clientFd, text: text)
        case 8: // Close Frame
            cancelTask(clientFd: clientFd)
            closeClient(clientFd)
        case 9: // Ping Frame
            sendWebSocketFrame(clientFd: clientFd, opcode: 10, payload: payload)
        default:
            break
        }
    }

    /// テキスト JSON コマンドの解析
    private func handleTextCommand(clientFd: Int32, text: String) {
        guard let data = text.data(using: .utf8) else { return }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            sendError(clientFd: clientFd, message: "JSON形式が不正です。")
            return
        }

        switch type {
        case "cancel":
            cancelTask(clientFd: clientFd)
        case "synthesize":
            handleSynthesize(clientFd: clientFd, data: data)
        default:
            sendError(clientFd: clientFd, message: "未対応のコマンドタイプです: \(type)")
        }
    }

    /// 進行中音声合成タスクの中断
    private func cancelTask(clientFd: Int32) {
        stateLock.lock()
        let token = clientTokens.removeValue(forKey: clientFd)
        stateLock.unlock()
        token?.cancel()
    }

    /// 音声合成リクエストの処理
    private func handleSynthesize(clientFd: Int32, data: Data) {
        let request: SynthesizeRequest
        do {
            request = try JSONDecoder().decode(SynthesizeRequest.self, from: data)
        } catch {
            sendError(clientFd: clientFd, message: "JSON形式が不正です: \(error.localizedDescription)")
            return
        }

        let synthText = request.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if synthText.isEmpty {
            sendError(clientFd: clientFd, message: "合成テキストが空です。")
            return
        }

        let token = CancellationToken()
        stateLock.lock()
        let oldToken = clientTokens[clientFd]
        clientTokens[clientFd] = token
        stateLock.unlock()
        oldToken?.cancel()

        let voiceName = request.voice ?? "female"
        let voiceProfile = VoiceProfile.preset(named: voiceName)
        let startTime = PlatformTime.uptimeNanoseconds()

        switch request.mode {
        case "wav":
            synthQueue.async { [weak self] in
                guard let self = self else { return }
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
                self.sendJSON(clientFd: clientFd, event: startEvent)

                if token.isCancelled {
                    return
                }

                self.sendWebSocketFrame(clientFd: clientFd, opcode: 2, payload: wavData)

                if token.isCancelled {
                    return
                }

                let doneEvent = ServerDoneEvent(type: "done", duration: audioDuration, samples: sampleCount, rtf: rtf)
                self.sendJSON(clientFd: clientFd, event: doneEvent)

                self.stateLock.lock()
                if self.clientTokens[clientFd] === token {
                    self.clientTokens.removeValue(forKey: clientFd)
                }
                self.stateLock.unlock()
            }

        default:
            // ストリーミング PCM 逐次配信モード
            let startEvent = ServerStartEvent(type: "start", sampleRate: Int(self.engine.sampleRate), channels: 1, format: "pcm_f32")
            sendJSON(clientFd: clientFd, event: startEvent)

            synthQueue.async { [weak self] in
                guard let self = self else { return }

                var totalSamples = 0
                let samples = self.engine.synthesizeStream(
                    text: synthText,
                    voice: voiceProfile,
                    speed: request.speed,
                    pitch: request.pitch,
                    isCancelled: {
                        token.isCancelled
                    },
                    onFrame: { [weak self] chunk in
                        guard let self = self else { return }
                        if token.isCancelled {
                            return
                        }
                        let byteCount = chunk.count * MemoryLayout<Float>.size
                        let chunkData = chunk.withUnsafeBufferPointer { buf in
                            Data(bytes: buf.baseAddress!, count: byteCount)
                        }
                        self.sendWebSocketFrame(clientFd: clientFd, opcode: 2, payload: chunkData)
                    }
                )

                if token.isCancelled {
                    self.stateLock.lock()
                    if self.clientTokens[clientFd] === token {
                        self.clientTokens.removeValue(forKey: clientFd)
                    }
                    self.stateLock.unlock()
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
                self.sendJSON(clientFd: clientFd, event: doneEvent)

                self.stateLock.lock()
                if self.clientTokens[clientFd] === token {
                    self.clientTokens.removeValue(forKey: clientFd)
                }
                self.stateLock.unlock()
            }
        }
    }

    /// エラーメッセージ送信
    private func sendError(clientFd: Int32, message: String) {
        let err = ServerErrorEvent(type: "error", message: message)
        sendJSON(clientFd: clientFd, event: err)
    }

    /// JSON イベント送信
    private func sendJSON<T: Encodable>(clientFd: Int32, event: T) {
        do {
            let data = try JSONEncoder().encode(event)
            sendWebSocketFrame(clientFd: clientFd, opcode: 1, payload: data)
        } catch {
            print("[POSIXWebServer] JSONエンコード失敗: \(error)")
        }
    }

    /// WebSocket フレームの送信 (RFC 6455 サーバー送信は非マスク)
    public func sendWebSocketFrame(clientFd: Int32, opcode: UInt8, payload: Data) {
        var frame = Data()
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
        sendRaw(clientFd: clientFd, data: frame)
    }

    /// ソケットへの同期バイト送信
    private func sendRaw(clientFd: Int32, data: Data) {
        data.withUnsafeBytes { rawBuf in
            var sent = 0
            let total = data.count
            let ptr = rawBuf.baseAddress!
            while sent < total {
                #if canImport(Darwin)
                let n = Darwin.send(clientFd, ptr.advanced(by: sent), total - sent, 0)
                #else
                let n = Glibc.send(clientFd, ptr.advanced(by: sent), total - sent, Int32(MSG_NOSIGNAL))
                #endif
                if n <= 0 {
                    break
                }
                sent += n
            }
        }
    }

    /// クライアント切断処理
    private func closeClient(_ clientFd: Int32) {
        stateLock.lock()
        let source = clientSources.removeValue(forKey: clientFd)
        clientBuffers.removeValue(forKey: clientFd)
        clientIsWebSocket.removeValue(forKey: clientFd)
        let token = clientTokens.removeValue(forKey: clientFd)
        stateLock.unlock()

        token?.cancel()
        source?.cancel()
    }
}
