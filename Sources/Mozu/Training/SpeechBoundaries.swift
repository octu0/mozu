import Foundation

/// 短時間 RMS による発話区間の検出
///
/// 録音の前後には任意長の無音があり、それを先頭・末尾の音素に割り当てると音響モデルが無音のスペクトルを学習してしまう。
/// 先頭・末尾の無音はそれぞれ独立した無音音素として扱う。
public enum SpeechBoundaries {
    public static func detect(pcm: [Float], hopSize: Int = AudioConfig.hopSize, totalFrames: Int) -> (leadSilence: Int, speechFrames: Int, trailSilence: Int) {
        if totalFrames <= 0 {
            return (0, 0, 0)
        }
        var frameRms = [Float](repeating: 0.0, count: totalFrames)
        var maxRms: Float = 0.0
        var f = 0
        while f < totalFrames {
            let start = f * hopSize
            var sumSq: Float = 0.0
            var count = 0
            var s = 0
            while s < hopSize {
                let idx = start + s
                if idx < pcm.count {
                    sumSq += pcm[idx] * pcm[idx]
                    count += 1
                }
                s += 1
            }
            let rms = 0 < count ? sqrtf(sumSq / Float(count)) : 0.0
            frameRms[f] = rms
            if maxRms < rms {
                maxRms = rms
            }
            f += 1
        }
        // 発話内ピーク RMS の 6% を無音／有音の境界とする
        let threshold = max(0.010, maxRms * 0.06)
        var firstSpeech = -1
        var lastSpeech = -1
        f = 0
        while f < totalFrames {
            if threshold <= frameRms[f] {
                if firstSpeech < 0 {
                    firstSpeech = f
                }
                lastSpeech = f
            }
            f += 1
        }
        if firstSpeech < 0 {
            return (0, totalFrames, 0)
        }
        let leadMargin = 5   // 語頭の弱い子音を守る 50 ms
        let trailMargin = 1  // 語尾の摩擦音を守る 10 ms
        let leadSilence = max(0, firstSpeech - leadMargin)
        let endFrame = min(totalFrames, lastSpeech + 1 + trailMargin)
        let speechFrames = max(1, endFrame - leadSilence)
        let trailSilence = max(0, totalFrames - (leadSilence + speechFrames))
        return (leadSilence, speechFrames, trailSilence)
    }
}
