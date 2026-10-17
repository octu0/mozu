import Foundation

/// 学習コーパスの 1 発話
public struct CorpusUtterance: Sendable, Equatable {
    public let id: String
    public let text: String
    public let wavPath: String
    /// ラベル（speaker, gender, age, tempo, tension, dialect など）。無ければ空
    public let labels: [String: String]

    public init(id: String, text: String, wavPath: String, labels: [String: String]) {
        self.id = id
        self.text = text
        self.wavPath = wavPath
        self.labels = labels
    }
}

/// 学習コーパス
///
/// ディレクトリ構成（コーパス固有の変換はリポジトリ外のスクリプトで行い、ライブラリはこの形式だけを知る）:
///   transcript_utf8.txt   "ID:本文" または "ID<TAB>本文" を 1 行 1 発話
///   wav/ID.wav            16 kHz モノラル（他のレートは読み込み時に変換）
///   labels.json           {"ID": {"speaker": "...", "gender": "...", ...}}（任意）
///   readings.json         {"表記": "読み（かな）", ...}（任意）。英字の語など、転写から読みが決まらない語の読み
///   mozu_alignments.json  音素アライメントのキャッシュ（align サブコマンドが作る）
public struct Corpus: Sendable {
    public let directory: String
    public let utterances: [CorpusUtterance]
    /// 表記 → 読み（readings.json、無ければ空）
    public let readings: [String: String]

    public static let transcriptFileName = "transcript_utf8.txt"
    public static let labelsFileName = "labels.json"
    public static let readingsFileName = "readings.json"
    public static let alignmentFileName = "mozu_alignments.json"

    public init(directory: String, utterances: [CorpusUtterance], readings: [String: String] = [:]) {
        self.directory = directory
        self.utterances = utterances
        self.readings = readings
    }

    public var alignmentPath: String {
        return directory + "/" + Self.alignmentFileName
    }

    public static func load(directory: String, limit: Int? = nil) throws -> Corpus {
        let transcriptPath = directory + "/" + transcriptFileName
        let content = try String(contentsOfFile: transcriptPath, encoding: .utf8)
        var labels: [String: [String: String]] = [:]
        let labelsPath = directory + "/" + labelsFileName
        if FileManager.default.fileExists(atPath: labelsPath) {
            let data = try Data(contentsOf: URL(fileURLWithPath: labelsPath))
            labels = try JSONDecoder().decode([String: [String: String]].self, from: data)
        }
        var readings: [String: String] = [:]
        let readingsPath = directory + "/" + readingsFileName
        if FileManager.default.fileExists(atPath: readingsPath) {
            let data = try Data(contentsOf: URL(fileURLWithPath: readingsPath))
            readings = try JSONDecoder().decode([String: String].self, from: data)
        }
        var utterances: [CorpusUtterance] = []
        let wavDir = directory + "/wav"
        for rawLine in content.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                continue
            }
            var parts = line.split(separator: ":", maxSplits: 1).map { String($0) }
            if parts.count != 2 {
                parts = line.split(separator: "\t", maxSplits: 1).map { String($0) }
            }
            if parts.count != 2 {
                continue
            }
            let id = parts[0].trimmingCharacters(in: .whitespaces)
            let text = parts[1].trimmingCharacters(in: .whitespaces)
            var wavPath = wavDir + "/" + id + ".wav"
            if FileManager.default.fileExists(atPath: wavPath) != true {
                let upper = wavDir + "/" + id + ".WAV"
                if FileManager.default.fileExists(atPath: upper) {
                    wavPath = upper
                } else {
                    continue
                }
            }
            utterances.append(CorpusUtterance(id: id, text: text, wavPath: wavPath, labels: labels[id] ?? [:]))
            if let limit = limit, limit <= utterances.count {
                break
            }
        }
        return Corpus(directory: directory, utterances: utterances, readings: readings)
    }

    /// 読み込み済み PCM（16 kHz、ピーク正規化済み）
    public static func loadPCM(path: String, reader: WavAudioReader) -> [Float]? {
        guard let raw = try? reader.loadWav16k(from: path), raw.isEmpty != true else {
            return nil
        }
        var peak: Float = 0.0
        for v in raw {
            let a = abs(v)
            if peak < a {
                peak = a
            }
        }
        if peak <= 0.01 {
            return raw
        }
        let scale = 0.85 / peak
        return raw.map { $0 * scale }
    }
}

/// 複数コーパスから条件付けラベルの語彙を集める
public extension LabelVocabulary {
    mutating func register(corpora: [Corpus]) {
        for corpus in corpora {
            for u in corpus.utterances {
                for (kind, value) in u.labels.sorted(by: { $0.key < $1.key }) {
                    register(kind: kind, value: value)
                }
            }
        }
    }
}
