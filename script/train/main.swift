import Foundation
import Mozu

/// train: mozu の学習 CLI（macOS / MLX）
///
/// サブコマンド:
///   align     コーパスの音素アライメント（MAS）を作り、コーパス内に保存する
///   acoustic  音響モデルを学習する
///   vocoder   ボコーダを学習する
///   lexicon   既存の重み（旧形式も可）から語彙を取り込む
///   styles    話し方の潜在変数を学習データで分類し、合成で選べる代表点を重みに保存する
func printUsage() {
    print("""
    Usage:
      train align    -d <corpus> [-d <corpus> ...]
      train acoustic -d <corpus> [-d <corpus> ...] -o <out.json> [-w <in.json>] [-e <epochs>] [--batch <n>] [--lr <lr>] [...]
      train vocoder  -d <corpus> [-d <corpus> ...] -o <out.json> [-w <in.json>] [-e <epochs>] [--batch <n>] [--lr <lr>] [...]
      train lexicon  --from <weights.json> -o <out.json> [-w <in.json>]
      train styles   -d <corpus> [-d <corpus> ...] -w <in.json> -o <out.json> [--clusters 12] [--samples 20000]
      train acoustic ... --style [--kl-weight 0.001] [--free-bits 0.1] [--kl-warmup 5000] [--style-dropout 0.2]   話し方の潜在変数つき
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
case "styles":
    exit(TrainCommands.runStyles(rest))
case "--help", "-h", "help":
    printUsage()
    exit(0)
default:
    print("不明なサブコマンド: \(command)")
    printUsage()
    exit(1)
}
