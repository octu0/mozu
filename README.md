<p align="center">
  <img src="assets/mozu-logo.svg" alt="mozu (鵙)" width="560">
</p>

# mozu

mozu (鵙, bull-headed shrike) is a Japanese text-to-speech (TTS) engine written in Swift for Apple Silicon.

## Usage

```
swift build -c release
.build/release/synthesize -w Models/weights.json -t "こんにちは、本日は晴天なり。" -o out.wav
.build/release/mozu-web -w Models/weights.json --port 8080
```

## Training

```
.build/release/train acoustic -d <corpus> -w Models/weights.json -o Models/weights.json -e 20
.build/release/train vocoder  -d <corpus> -w Models/weights.json -o Models/weights.json -e 50 --gan
```

A corpus is a directory with `transcript_utf8.txt` (`ID:text` per line), `wav/ID.wav`, and an optional `labels.json`.

## License

MIT. See [LICENSE](LICENSE).
