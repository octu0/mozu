import Foundation

/// ブラウザで直接 Mozu の合成音声を試聴・確認するための内蔵 Web UI
public enum WebClientHTML {

    /// 単一ファイルで完結するモダンな HTML/CSS/JavaScript ページ文字列
    public static let content: String = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mozu - SNN 音声合成エンジン</title>
        <style>
            :root {
                --bg: #0f172a;
                --surface: #1e293b;
                --surface-hover: #334155;
                --primary: #38bdf8;
                --primary-glow: rgba(56, 189, 248, 0.3);
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --border: #334155;
                --accent: #a855f7;
                --success: #10b981;
                --error: #ef4444;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                background: var(--bg);
                color: var(--text);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 2rem 1rem;
            }
            .container {
                width: 100%;
                max-width: 800px;
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
            }
            header {
                margin-bottom: 2rem;
                text-align: center;
            }
            h1 {
                font-size: 1.8rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                background: linear-gradient(135deg, var(--primary), var(--accent));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                font-size: 0.95rem;
                color: var(--text-muted);
            }
            .form-group {
                margin-bottom: 1.5rem;
            }
            label {
                display: block;
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--text-muted);
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            textarea {
                width: 100%;
                height: 110px;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--text);
                padding: 0.75rem;
                font-size: 1rem;
                line-height: 1.5;
                resize: vertical;
                outline: none;
                transition: border-color 0.2s;
            }
            textarea:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px var(--primary-glow);
            }
            .controls-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .slider-container {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            .slider-header {
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                color: var(--text-muted);
            }
            input[type="range"] {
                width: 100%;
                accent-color: var(--primary);
            }
            .mode-selector {
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            .mode-option {
                flex: 1;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--border);
                padding: 0.75rem;
                border-radius: 8px;
                cursor: pointer;
                user-select: none;
                font-size: 0.9rem;
            }
            .mode-option input {
                accent-color: var(--primary);
            }
            .btn-group {
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            button {
                flex: 1;
                padding: 0.9rem 1.5rem;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                border: none;
                transition: all 0.2s;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 0.5rem;
            }
            .btn-primary {
                background: linear-gradient(135deg, var(--primary), #0284c7);
                color: #0f172a;
            }
            .btn-primary:hover:not(:disabled) {
                filter: brightness(1.1);
                box-shadow: 0 0 15px var(--primary-glow);
            }
            .btn-primary:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .btn-secondary {
                background: var(--surface-hover);
                color: var(--text);
                border: 1px solid var(--border);
            }
            .btn-secondary:hover:not(:disabled) {
                background: #475569;
            }
            .visualizer-card {
                background: rgba(15, 23, 42, 0.9);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1.5rem;
            }
            canvas {
                width: 100%;
                height: 80px;
                display: block;
                border-radius: 6px;
            }
            .status-panel {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 0.75rem;
                text-align: center;
            }
            .stat-box {
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid var(--border);
                padding: 0.5rem;
                border-radius: 8px;
            }
            .stat-label {
                font-size: 0.7rem;
                color: var(--text-muted);
                text-transform: uppercase;
                margin-bottom: 0.2rem;
            }
            .stat-val {
                font-size: 0.95rem;
                font-weight: 700;
                color: var(--primary);
                font-variant-numeric: tabular-nums;
            }
            .connection-status {
                margin-top: 1rem;
                text-align: center;
                font-size: 0.8rem;
                color: var(--text-muted);
            }
            .badge {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 0.4rem;
            }
            .badge-connected { background: var(--success); box-shadow: 0 0 8px var(--success); }
            .badge-disconnected { background: var(--error); }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Mozu Web</h1>
                <p class="subtitle">Pure Swift + 多層 SNN による超低遅延・高音質日本語音声合成</p>
            </header>

            <div class="form-group">
                <label for="text-input">合成テキスト</label>
                <textarea id="text-input">こんにちは。スパイキングニューラルネットワークによる超低遅延音声合成の世界へようこそ。</textarea>
            </div>

            <div class="controls-grid">
                <div class="slider-container">
                    <div class="slider-header">
                        <span>話速 (Speed)</span>
                        <span id="speed-val">1.0x</span>
                    </div>
                    <input type="range" id="speed-input" min="0.5" max="2.0" step="0.1" value="1.0">
                </div>
                <div class="slider-container">
                    <div class="slider-header">
                        <span>ピッチ (Pitch)</span>
                        <span id="pitch-val">1.0x</span>
                    </div>
                    <input type="range" id="pitch-input" min="0.5" max="2.0" step="0.1" value="1.0">
                </div>
            </div>

            <div class="form-group" style="margin-bottom: 1.5rem;">
                <label for="voice-select">話者・声質 (Voice Profile)</label>
                <select id="voice-select" style="width: 100%; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 0.6rem; font-size: 0.95rem; outline: none;">
                    <option value="female" selected>女性 (gender=female)</option>
                    <option value="male">男性 (gender=male)</option>
                    <option value="neutral">指定なし (学習データの平均)</option>
                    <option value="gender=female,tempo=ゆっくり,tension=落ち着いてる">女性・ゆっくり・落ち着き</option>
                    <option value="gender=female,tempo=早口,tension=ハイテンション">女性・早口・ハイテンション</option>
                    <option value="gender=male,tempo=ゆっくり,tension=落ち着いてる">男性・ゆっくり・落ち着き</option>
                    <option value="gender=female,dialect=関西弁 (大阪)">女性・関西弁</option>
                </select>
            </div>

            <div class="mode-selector">
                <label class="mode-option">
                    <input type="radio" name="mode" value="stream" checked>
                    <span>ストリーミング (PCM 逐次低遅延再生)</span>
                </label>
                <label class="mode-option">
                    <input type="radio" name="mode" value="wav">
                    <span>WAV (一括バイナリ)</span>
                </label>
            </div>

            <div class="btn-group">
                <button id="btn-play" class="btn-primary">
                    <span id="play-icon">▶</span>
                    <span id="play-text">合成して再生</span>
                </button>
                <button id="btn-stop" class="btn-secondary" disabled>停止</button>
            </div>

            <div class="visualizer-card">
                <canvas id="waveform-canvas"></canvas>
            </div>

            <div class="status-panel">
                <div class="stat-box">
                    <div class="stat-label">首尾一貫性 (RTF)</div>
                    <div class="stat-val" id="stat-rtf">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">音声長</div>
                    <div class="stat-val" id="stat-dur">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">サンプル数</div>
                    <div class="stat-val" id="stat-samples">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">応答時間</div>
                    <div class="stat-val" id="stat-time">-</div>
                </div>
            </div>

            <div class="connection-status">
                <span class="badge badge-disconnected" id="conn-badge"></span>
                <span id="conn-text">接続中...</span>
            </div>
        </div>

        <script>
            let ws = null;
            let audioCtx = null;
            let masterGainNode = null;
            let streamScheduledTime = 0;
            let activeSourceNodes = [];
            let isSynthesizing = false;
            let currentRequestId = 0;
            let activePlaybackRequestId = 0;
            let requestStartTime = 0;

            const textInput = document.getElementById('text-input');
            const speedInput = document.getElementById('speed-input');
            const pitchInput = document.getElementById('pitch-input');
            const speedVal = document.getElementById('speed-val');
            const pitchVal = document.getElementById('pitch-val');
            const btnPlay = document.getElementById('btn-play');
            const btnStop = document.getElementById('btn-stop');
            const playText = document.getElementById('play-text');
            const connBadge = document.getElementById('conn-badge');
            const connText = document.getElementById('conn-text');
            const statRtf = document.getElementById('stat-rtf');
            const statDur = document.getElementById('stat-dur');
            const statSamples = document.getElementById('stat-samples');
            const statTime = document.getElementById('stat-time');
            const canvas = document.getElementById('waveform-canvas');
            const canvasCtx = canvas.getContext('2d');

            speedInput.addEventListener('input', (e) => {
                speedVal.textContent = parseFloat(e.target.value).toFixed(1) + 'x';
            });
            pitchInput.addEventListener('input', (e) => {
                pitchVal.textContent = parseFloat(e.target.value).toFixed(1) + 'x';
            });

            function resizeCanvas() {
                canvas.width = canvas.clientWidth * window.devicePixelRatio;
                canvas.height = canvas.clientHeight * window.devicePixelRatio;
                drawEmptyWaveform();
            }
            window.addEventListener('resize', resizeCanvas);
            setTimeout(resizeCanvas, 50);

            function drawEmptyWaveform() {
                canvasCtx.fillStyle = '#0f172a';
                canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                canvasCtx.strokeStyle = '#334155';
                canvasCtx.lineWidth = 1 * window.devicePixelRatio;
                canvasCtx.beginPath();
                canvasCtx.moveTo(0, canvas.height / 2);
                canvasCtx.lineTo(canvas.width, canvas.height / 2);
                canvasCtx.stroke();
            }

            function drawWaveform(samples) {
                canvasCtx.fillStyle = '#0f172a';
                canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

                canvasCtx.strokeStyle = '#38bdf8';
                canvasCtx.lineWidth = 1.5 * window.devicePixelRatio;
                canvasCtx.beginPath();

                const sliceWidth = canvas.width / samples.length;
                let x = 0;
                var i = 0;
                while (i < samples.length) {
                    const v = samples[i];
                    const y = (0.5 - (v * 0.45)) * canvas.height;
                    if (i === 0) {
                        canvasCtx.moveTo(x, y);
                    } else {
                        canvasCtx.lineTo(x, y);
                    }
                    x += sliceWidth;
                    i += 1;
                }
                canvasCtx.stroke();
            }

            function initAudio() {
                if (!audioCtx) {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                }
                if (!masterGainNode) {
                    masterGainNode = audioCtx.createGain();
                    masterGainNode.connect(audioCtx.destination);
                }
                if (audioCtx.state === 'suspended') {
                    audioCtx.resume();
                }
            }

            function connectWebSocket() {
                var wsProtocol = 'ws:';
                if (window.location.protocol === 'https:') {
                    wsProtocol = 'wss:';
                }
                const host = window.location.host;
                const url = wsProtocol + '//' + host + '/ws';

                ws = new WebSocket(url);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    connBadge.className = 'badge badge-connected';
                    connText.textContent = 'サーバー接続中 (オンライン)';
                    btnPlay.disabled = false;
                };

                ws.onclose = () => {
                    connBadge.className = 'badge badge-disconnected';
                    connText.textContent = '接続切断 - 3秒後に再試行';
                    btnPlay.disabled = true;
                    setTimeout(connectWebSocket, 3000);
                };

                ws.onerror = (err) => {
                    console.error('WebSocket Error:', err);
                };

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        try {
                            const msg = JSON.parse(event.data);
                            handleServerMessage(msg);
                        } catch (e) {
                            console.error('Failed to parse JSON:', e);
                        }
                    }
                    if (event.data instanceof ArrayBuffer) {
                        handleBinaryAudio(event.data);
                    }
                };
            }

            let currentMode = 'stream';
            let collectedSamples = [];

            function handleServerMessage(msg) {
                switch (msg.type) {
                case 'start':
                    if (isSynthesizing != true) {
                        return;
                    }
                    initAudio();
                    if (masterGainNode) {
                        masterGainNode.gain.setValueAtTime(1.0, audioCtx.currentTime);
                    }
                    streamScheduledTime = audioCtx.currentTime + 0.05;
                    collectedSamples = [];
                    break;
                case 'done':
                    const totalTimeMs = Date.now() - requestStartTime;
                    statRtf.textContent = msg.rtf.toFixed(4);
                    statDur.textContent = msg.duration.toFixed(2) + 's';
                    statSamples.textContent = msg.samples.toLocaleString();
                    statTime.textContent = totalTimeMs + 'ms';
                    setPlayingState(false);
                    if (0 < collectedSamples.length) {
                        drawWaveform(collectedSamples);
                    }
                    break;
                case 'error':
                    alert('エラー: ' + msg.message);
                    stopAllAudio();
                    break;
                }
            }

            async function handleBinaryAudio(arrayBuffer) {
                if (currentRequestId != activePlaybackRequestId) {
                    return;
                }

                const myRequestId = activePlaybackRequestId;

                switch (currentMode) {
                case 'wav':
                    initAudio();
                    try {
                        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
                        // 途中でキャンセルや新規リクエストが発行されていない限り確実に再生
                        if (currentRequestId != myRequestId) {
                            return;
                        }

                        const source = audioCtx.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(masterGainNode);
                        source.start(0);
                        activeSourceNodes.push(source);

                        source.onended = () => {
                            var idx = activeSourceNodes.indexOf(source);
                            if (0 <= idx) {
                                activeSourceNodes.splice(idx, 1);
                            }
                        };

                        const channelData = audioBuffer.getChannelData(0);
                        const step = Math.ceil(channelData.length / 1000);
                        const sampled = [];
                        var cIdx = 0;
                        while (cIdx < channelData.length) {
                            sampled.push(channelData[cIdx]);
                            cIdx += step;
                        }
                        drawWaveform(sampled);
                    } catch (e) {
                        console.error('Audio decode error:', e);
                    }
                    break;
                default:
                    const float32Array = new Float32Array(arrayBuffer);
                    var fIdx = 0;
                    while (fIdx < float32Array.length) {
                        collectedSamples.push(float32Array[fIdx]);
                        fIdx += 4;
                    }

                    const audioBuffer = audioCtx.createBuffer(1, float32Array.length, 16000);
                    audioBuffer.getChannelData(0).set(float32Array);

                    const source = audioCtx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(masterGainNode);

                    if (streamScheduledTime < audioCtx.currentTime) {
                        streamScheduledTime = audioCtx.currentTime;
                    }
                    source.start(streamScheduledTime);
                    streamScheduledTime += audioBuffer.duration;
                    activeSourceNodes.push(source);

                    source.onended = () => {
                        var idx = activeSourceNodes.indexOf(source);
                        if (0 <= idx) {
                            activeSourceNodes.splice(idx, 1);
                        }
                    };
                    break;
                }
            }

            function setPlayingState(playing) {
                isSynthesizing = playing;
                btnPlay.disabled = playing;
                btnStop.disabled = (playing != true);
                if (playing) {
                    playText.textContent = '合成中...';
                } else {
                    playText.textContent = '合成して再生';
                }
            }

            btnPlay.addEventListener('click', () => {
                const text = textInput.value.trim();
                if (!text) {
                    alert('合成テキストを入力してください。');
                    return;
                }
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('サーバーに接続されていません。');
                    return;
                }

                stopAllAudio();

                currentRequestId += 1;
                activePlaybackRequestId = currentRequestId;

                initAudio();
                if (masterGainNode) {
                    masterGainNode.gain.setValueAtTime(1.0, audioCtx.currentTime);
                }

                const mode = document.querySelector('input[name="mode"]:checked').value;
                currentMode = mode;
                setPlayingState(true);
                requestStartTime = Date.now();

                const voiceSelect = document.getElementById('voice-select');
                let voiceVal = 'female';
                if (voiceSelect) {
                    voiceVal = voiceSelect.value;
                }

                const payload = {
                    type: 'synthesize',
                    text: text,
                    voice: voiceVal,
                    speed: parseFloat(speedInput.value),
                    pitch: parseFloat(pitchInput.value),
                    mode: mode
                };
                ws.send(JSON.stringify(payload));
            });

            function stopAllAudio() {
                currentRequestId += 1;
                isSynthesizing = false;

                if (masterGainNode) {
                    if (audioCtx) {
                        masterGainNode.gain.setValueAtTime(0.0, audioCtx.currentTime);
                    }
                }

                var nIdx = 0;
                while (nIdx < activeSourceNodes.length) {
                    var node = activeSourceNodes[nIdx];
                    try {
                        node.stop(0);
                        node.disconnect();
                    } catch (e) {}
                    nIdx += 1;
                }
                activeSourceNodes = [];
                streamScheduledTime = 0;

                setPlayingState(false);

                if (ws) {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'cancel' }));
                    }
                }
            }

            btnStop.addEventListener('click', stopAllAudio);

            connectWebSocket();
        </script>
    </body>
    </html>
    """
}
