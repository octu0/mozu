import Foundation
import Mozu

/// train: mozu の学習 CLI（macOS / MLX）
///
/// サブコマンド:
///   align     コーパスの音素アライメント（MAS）を作り、コーパス内に保存する
///   acoustic  音響モデルを学習する
///   vocoder   ボコーダを学習する
///   lexicon   既存の重み（旧形式も可）から語彙を取り込む
func printUsage() {
    print("""
    Usage:
      train align    -d <corpus> [-d <corpus> ...]
      train acoustic -d <corpus> [-d <corpus> ...] -o <out.json> [-w <in.json>] [-e <epochs>] [--batch <n>] [--lr <lr>] [...]
      train vocoder  -d <corpus> [-d <corpus> ...] -o <out.json> [-w <in.json>] [-e <epochs>] [--batch <n>] [--lr <lr>] [...]
      train lexicon  --from <weights.json> -o <out.json> [-w <in.json>]
    """)
}

// ファイルへリダイレクトしても進捗が逐次書かれるよう行バッファにする（落ちたときに末尾が消えない）
setvbuf(stdout, nil, _IOLBF, 0)

let arguments = Array(CommandLine.arguments.dropFirst())
guard let command = arguments.first else {
    printUsage()
    exit(1)
}
let rest = Array(arguments.dropFirst())
switch command {
case "align":
    exit(TrainCommands.runAlign(rest))
case "acoustic":
    exit(TrainCommands.runAcoustic(rest))
case "vocoder":
    exit(TrainCommands.runVocoder(rest))
case "lexicon":
    exit(TrainCommands.runLexicon(rest))
case "--help", "-h", "help":
    printUsage()
    exit(0)
default:
    print("不明なサブコマンド: \(command)")
    printUsage()
    exit(1)
}
