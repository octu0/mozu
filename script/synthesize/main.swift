import Foundation
import Mozu

/// synthesize: テキストから WAV を合成する CLI
///
/// 例: synthesize -w Models/weights.json -t "こんにちは" -o out.wav
///     synthesize -w Models/weights.json --copy input.wav -o copy.wav   （Mel を抽出してボコーダだけを通す）
func printUsage() {
    print("""
    Usage: synthesize -w <weights.json> (-t <text> | --copy <input.wav>) [-o <output.wav>] [-v <voice>] [--speed 1.0] [--pitch 1.0] [--mel-out <mel.json>]
      -v  プリセット名 (female/male/neutral) または key=value,key=value（例: gender=female,dialect=標準語）
    """)
}

var weightsPath: String? = nil
var text: String? = nil
var copyPath: String? = nil
var outputPath = "output.wav"
var voiceName = "default"
var speed: Float = 1.0
var pitch: Float = 1.0
var melOutPath: String? = nil

let args = CommandLine.arguments
var i = 1
while i < args.count {
    let a = args[i]
    func next() -> String? {
        if i + 1 < args.count {
            i += 1
            return args[i]
        }
        return nil
    }
    switch a {
    case "-w", "--weights":
        weightsPath = next()
    case "-t", "--text":
        text = next()
    case "--copy":
        copyPath = next()
    case "-o", "--output":
        outputPath = next() ?? outputPath
    case "-v", "--voice":
        voiceName = next() ?? voiceName
    case "--speed":
        speed = Float(next() ?? "1") ?? 1.0
    case "--pitch":
        pitch = Float(next() ?? "1") ?? 1.0
    case "--mel-out":
        melOutPath = next()
    case "--help", "-h":
        printUsage()
        exit(0)
    default:
        print("不明な引数: \(a)")
        printUsage()
        exit(1)
    }
    i += 1
}

guard let wPath = weightsPath else {
    print("エラー: -w <weights.json> を指定してください。")
    printUsage()
    exit(1)
}
let engine: MozuEngine
do {
    engine = try MozuEngine.load(from: wPath)
} catch {
    print("エラー: 重みを読み込めません: \(error)")
    exit(1)
}
let voice = VoiceProfile.preset(named: voiceName)
var samples: [Float] = []
let started = Date()
if let input = copyPath {
    guard engine.vocoder != nil else {
        print("エラー: ボコーダの重みがありません。")
        exit(1)
    }
    do {
        let pcm = try WavAudioReader().loadWav16k(from: input)
        let mel = MelSpectrogramExtractor().extractLogMel(pcm: pcm)
        samples = engine.vocode(mel: mel)
        print("コピー合成: \(mel.count) フレーム → \(samples.count) サンプル")
    } catch {
        print("エラー: 入力 WAV を読めません: \(error)")
        exit(1)
    }
} else if let t = text {
    guard engine.acoustic != nil else {
        print("エラー: 音響モデルの重みがありません。")
        exit(1)
    }
    if let result = engine.synthesizeMel(text: t, voice: voice, speed: speed, pitch: pitch) {
        let symbols = result.sequence.tokens.map { $0.symbol }
        let durations = result.output.durations
        var line = ""
        var k = 0
        while k < symbols.count {
            let hz = result.sequence.tokens[k].category == .pause ? "-" : String(format: "%.0f", 220.0 * pow(2.0, result.output.pitch[k]))
            line += "\(symbols[k])(\(durations[k]),\(hz)) "
            k += 1
        }
        print("音素列 (フレーム数, F0 Hz): " + line)
        print("Mel \(result.output.mel.count) フレーム")
        if let melPath = melOutPath {
            if let data = try? JSONEncoder().encode(result.output.mel) {
                try? data.write(to: URL(fileURLWithPath: melPath))
            }
        }
        if engine.vocoder != nil {
            samples = engine.vocode(mel: result.output.mel)
        } else {
            print("ボコーダの重みが無いので WAV は書き出しません。")
            exit(0)
        }
    } else {
        print("エラー: 合成できませんでした（音素列が空か、モデルがありません）。")
        exit(1)
    }
} else {
    print("エラー: -t <text> または --copy <input.wav> を指定してください。")
    printUsage()
    exit(1)
}
let elapsed = Date().timeIntervalSince(started)
let data = WavEncoder.encode(samples: samples, sampleRate: AudioConfig.sampleRate)
do {
    try data.write(to: URL(fileURLWithPath: outputPath))
    let seconds = Float(samples.count) / Float(AudioConfig.sampleRate)
    print(String(format: "WAV を出力しました: %@ (%.2f 秒, 処理 %.2f 秒, RTF %.3f)", outputPath, seconds, elapsed, Float(elapsed) / max(0.01, seconds)))
} catch {
    print("エラー: 書き出しに失敗しました: \(error)")
    exit(1)
}
