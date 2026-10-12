import Foundation
import Mozu
import MozuWeb

/// mozu-web CLI
///
/// Mozu WebSocket & HTTP 音声合成サーバーを起動するコマンドラインツール。

func printUsage() {
    print("""
    Usage: mozu-web -w <weights.json> [-p <port>] [-h <host>]

    Options:
      -w, --weights <path>        mozu 重み JSON ファイルパス (必須)
      -p, --port <port>           バインドポート (デフォルト: 8080)
      -h, --host <host>           バインドホスト (デフォルト: 0.0.0.0)
      --help                      このヘルプを表示
    """)
}

var port: UInt16 = 8080
var host: String = "0.0.0.0"
var weightsPath: String?

// Cloud Run 等の環境変数 PORT を優先フォールバックとして取得
if let envPort = ProcessInfo.processInfo.environment["PORT"], let p = UInt16(envPort) {
    port = p
}

let args = CommandLine.arguments
var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-p", "--port":
        if i + 1 < args.count {
            i += 1
            if let p = UInt16(args[i]) {
                port = p
            }
        }
    case "-h", "--host":
        if i + 1 < args.count {
            i += 1
            host = args[i]
        }
    case "-w", "--weights":
        if i + 1 < args.count {
            i += 1
            weightsPath = args[i]
        }
    case "--help":
        printUsage()
        exit(0)
    default:
        break
    }
    i += 1
}

// 重みファイルパスの解決: 引数 -> 環境変数 WEIGHTS_PATH -> デフォルト Models/weights.json の順にフォールバック
var resolvedWeightsPath = weightsPath
if resolvedWeightsPath == nil {
    if let envPath = ProcessInfo.processInfo.environment["WEIGHTS_PATH"], envPath.isEmpty != true {
        resolvedWeightsPath = envPath
    } else {
        let defaultPath = "Models/weights.json"
        if FileManager.default.fileExists(atPath: defaultPath) {
            resolvedWeightsPath = defaultPath
        }
    }
}

// 疑似値やランダム重みへのフォールバックを排除し、実在する学習済み重みファイルの指定を必須とする
guard let path = resolvedWeightsPath else {
    print("エラー: 重みファイル (-w / --weights <path> または環境変数 WEIGHTS_PATH) の指定は必須です。")
    print("例: mozu-web -w Models/weights.json --port 8080")
    print("ヘルプ表示: mozu-web --help")
    exit(1)
}

let engine: MozuEngine
do {
    engine = try MozuEngine.load(from: path)
    print("[MozuWeb] 学習済み重みファイルを読み込みました: \(path)")
} catch {
    print("エラー: 重みファイル '\(path)' の読み込みまたは JSON 解析に失敗しました: \(error)")
    exit(1)
}
if engine.acoustic == nil || engine.vocoder == nil {
    print("警告: 音響モデルまたはボコーダの重みが含まれていません。合成結果は無音になります。")
}
let server = MozuWebServer(port: port, host: host, engine: engine)

do {
    try server.start()
    print("==================================================")
    print("Mozu Web サーバーが起動しました")
    print("==================================================")
    var displayHost = host
    if host == "0.0.0.0" {
        displayHost = "localhost"
    }
    print("URL:           http://\(displayHost):\(port)/")
    print("WebSocket:     ws://\(displayHost):\(port)/ws")
    print("サンプリング:  16,000 Hz (16-bit Mono)")
    print("パラメータ数:  \(engine.weights.parameterCount)")
    print("--------------------------------------------------")
    print("ブラウザで上記 URL にアクセスすると、GUI から音声を試聴できます。")
    print("Ctrl+C で停止します。")
    print("--------------------------------------------------")

    // シグナル待機でイベントループを維持
    signal(SIGINT) { _ in
        print("\n[MozuWeb] サーバーを終了します...")
        exit(0)
    }
    signal(SIGTERM) { _ in
        print("\n[MozuWeb] サーバーを終了します...")
        exit(0)
    }

    dispatchMain()
} catch {
    print("[MozuWeb] サーバーの起動に失敗しました: \(error)")
    exit(1)
}
