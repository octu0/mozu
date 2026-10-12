import XCTest
@testable import Mozu
@testable import MozuWeb

/// Web 領域（プロトコル JSON 型, WebClientHTML）の単体テストスイート
final class WebTests: XCTestCase {

    /// プロトコル JSON メッセージの相互エンコード・デコード検証
    func testProtocolJSONSerialization() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // 1. SynthesizeRequest
        let req = SynthesizeRequest(type: "synthesize", text: "テスト発話", speed: 1.2, pitch: 0.9, mode: "stream")
        let reqData = try encoder.encode(req)
        let decodedReq = try decoder.decode(SynthesizeRequest.self, from: reqData)
        XCTAssertEqual(decodedReq.text, "テスト発話")
        XCTAssertEqual(decodedReq.speed, 1.2)
        XCTAssertEqual(decodedReq.pitch, 0.9)
        XCTAssertEqual(decodedReq.mode, "stream")

        // 2. ServerStartEvent
        let startEvt = ServerStartEvent(type: "start", sampleRate: 16000, channels: 1, format: "pcm_f32")
        let startData = try encoder.encode(startEvt)
        let decodedStart = try decoder.decode(ServerStartEvent.self, from: startData)
        XCTAssertEqual(decodedStart.sampleRate, 16000)
        XCTAssertEqual(decodedStart.format, "pcm_f32")

        // 3. ServerDoneEvent
        let doneEvt = ServerDoneEvent(type: "done", duration: 1.5, samples: 24000, rtf: 0.015)
        let doneData = try encoder.encode(doneEvt)
        let decodedDone = try decoder.decode(ServerDoneEvent.self, from: doneData)
        XCTAssertEqual(decodedDone.samples, 24000)
        XCTAssertEqual(decodedDone.rtf, 0.015)

        // 4. ServerErrorEvent
        let errEvt = ServerErrorEvent(type: "error", message: "エラーメッセージ")
        let errData = try encoder.encode(errEvt)
        let decodedErr = try decoder.decode(ServerErrorEvent.self, from: errData)
        XCTAssertEqual(decodedErr.message, "エラーメッセージ")

        // 5. CancelRequest
        let cancelReq = CancelRequest(type: "cancel")
        let cancelData = try encoder.encode(cancelReq)
        let decodedCancel = try decoder.decode(CancelRequest.self, from: cancelData)
        XCTAssertEqual(decodedCancel.type, "cancel")
    }

    /// 内蔵 Web UI の HTML/CSS/JS 構文整合性検証
    func testWebClientHTMLIntegrity() {
        let html = WebClientHTML.content
        XCTAssertTrue(0 < html.count, "WebClientHTML が空であってはならない")
        XCTAssertTrue(html.contains("<!DOCTYPE html>"), "DOCTYPE宣言を含むこと")
        XCTAssertTrue(html.contains("Mozu Web"), "タイトル表記を含むこと")
        XCTAssertTrue(html.contains("AudioContext"), "Web Audio API 初期化コードを含むこと")
        XCTAssertTrue(html.contains("new WebSocket"), "WebSocket 接続コードを含むこと")
        XCTAssertTrue(html.contains("waveform-canvas"), "波形可視化 Canvas を含むこと")
    }
}
