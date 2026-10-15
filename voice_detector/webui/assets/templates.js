export const APP_ROUTES = {
    home: "/app",
    workspace: "/app/workspace",
    processing: "/app/processing",
    report: "/app/report",
    diagnostics: "/app/diagnostics",
    repository: "/app/repository",
    status: "/app/status",
};

function navButton(label, route, active = false) {
    return `<button class="nav-link${active ? " active" : ""}" data-route="${route}">${label}</button>`;
}

function globalTopbar(activeScreen) {
    const isActive = (screen) => activeScreen === screen;
    return `
        <header class="topbar topbar-app">
            <div class="brand" data-route="${APP_ROUTES.home}" style="cursor:pointer">VoxTrust</div>
            <nav class="nav-inline">
                ${navButton("Home", APP_ROUTES.home, isActive("home"))}
                ${navButton("Analyze", APP_ROUTES.workspace, isActive("workspace"))}
                ${navButton("Repository", APP_ROUTES.repository, isActive("repository"))}
                ${navButton("Report", APP_ROUTES.report, isActive("report"))}
                ${navButton("Status", APP_ROUTES.status, isActive("status"))}
            </nav>
            <div class="topbar-actions">
                <button class="btn btn-solid" data-open-uploader="true">New Scan</button>
            </div>
        </header>
    `;
}

function topbar({ brand, links, actions = "", className = "topbar topbar-app" }) {
    return `
        <header class="${className}">
            <div class="brand">${brand}</div>
            <nav class="nav-inline">
                ${links.join("")}
            </nav>
            <div class="topbar-actions">
                ${actions}
            </div>
        </header>
    `;
}

export function renderScreen(screen) {
    const templates = {
        home: homeScreen,
        workspace: workspaceScreen,
        processing: processingScreen,
        report: reportScreen,
        diagnostics: diagnosticsScreen,
        repository: repositoryScreen,
        status: statusScreen,
    };

    const template = templates[screen] || homeScreen;
    return `
        <input
            id="audio-input"
            class="sr-only"
            type="file"
            accept=".mp3,.wav,.flac,audio/mpeg,audio/wav,audio/x-wav,audio/flac"
        >
        <audio id="uploaded-audio" preload="metadata"></audio>
        ${template()}
        <footer class="site-footer">
            <span>Created by <strong>Mehul Gupta</strong></span>
        </footer>
    `;
}

function homeScreen() {
    return `
        <section class="ui-screen screen-home vt-reference-home">
            <header class="reference-topbar">
                <button class="reference-brand" data-route="${APP_ROUTES.home}" aria-label="VoxTrust home">
                    <span class="reference-brand-mark">▥</span>
                    <span><strong>VoxTrust</strong><small>AI Voice Detector</small></span>
                </button>
                <nav class="reference-topnav">
                    ${navButton("Home", APP_ROUTES.home, true)}
                    ${navButton("Analyze", APP_ROUTES.workspace)}
                    ${navButton("History", APP_ROUTES.repository)}
                    ${navButton("Diagnostics", APP_ROUTES.diagnostics)}
                    ${navButton("System Status", APP_ROUTES.status)}
                </nav>
                <div class="reference-top-actions">
                    <button class="reference-theme" type="button" aria-label="Theme preview"><span>☼</span><i></i><span>☾</span></button>
                    <button class="reference-avatar" data-route="${APP_ROUTES.status}" aria-label="Profile">M⌄</button>
                </div>
            </header>
            <aside class="reference-sidebar">
                <nav class="reference-side-nav">
                    <button class="active" data-route="${APP_ROUTES.home}"><span>⌂</span>Home</button>
                    <button data-route="${APP_ROUTES.workspace}"><span>♬</span>Analyze Audio</button>
                    <button data-route="${APP_ROUTES.repository}"><span>◷</span>Scan History</button>
                    <button data-route="${APP_ROUTES.diagnostics}"><span>▥</span>Diagnostics</button>
                    <button data-route="${APP_ROUTES.status}"><span>⛨</span>System Status</button>
                    <button data-route="${APP_ROUTES.status}"><span>ⓘ</span>About</button>
                </nav>
                <div class="reference-side-art"><span class="side-orb"></span><span class="side-shield">♬</span><small>Detect.<br>Verify.<br>Stay Ahead.</small><em>Powered by<br>Advanced AI</em><b>→</b></div>
            </aside>
            <main class="reference-main">
                <section class="reference-hero">
                    <div class="reference-hero-copy">
                        <span class="reference-eyebrow">TRUST IN EVERY VOICE</span>
                        <h1>Detect AI-Generated<br>Voices with <em>Confidence</em></h1>
                        <p>Upload an audio recording and let our AI analyze whether<br class="desktop-break"> it's a real human voice or an AI-generated voice.</p>
                        <div class="reference-benefits">
                            <div><span class="benefit-icon">✓</span><strong>High Accuracy<small>Advanced ML models</small></strong></div>
                            <div><span class="benefit-icon">ϟ</span><strong>Fast Analysis<small>Results in seconds</small></strong></div>
                            <div><span class="benefit-icon">♙</span><strong>Private &amp; Secure<small>Your data stays safe</small></strong></div>
                        </div>
                    </div>
                    <div class="reference-hero-art" aria-hidden="true">
                        <div class="reference-orbit orbit-one"></div><div class="reference-orbit orbit-two"></div>
                        <div class="reference-float-card human-card"><b>Human Voice</b><span class="mini-wave blue-wave"></span></div>
                        <div class="reference-mic"><span></span><i></i><b></b></div>
                        <div class="reference-float-card ai-card"><b>AI Generated</b><span class="mini-wave red-wave"></span></div>
                        <div class="reference-scan-chip">◉ &nbsp; Analyzing...</div>
                        <span class="reference-bubble bubble-one"></span><span class="reference-bubble bubble-two"></span><span class="reference-bubble bubble-three"></span>
                    </div>
                </section>
                <section class="reference-workbench">
                    <article class="reference-card upload-card">
                        <div class="reference-card-title"><h2><span>↥</span>Upload Audio</h2><span class="format-chip">◉ &nbsp; Supported formats</span></div>
                        <label class="reference-dropzone" id="dropzone" for="audio-input">
                            <span class="reference-upload-cloud">↥</span>
                            <strong>Drag and drop an audio file here</strong>
                            <span>or click to browse</span>
                            <span class="reference-upload-button">↥ &nbsp; Choose Audio File</span>
                            <small>Supported: WAV, MP3, M4A, FLAC, OGG &nbsp; • &nbsp; Max file size: 10 MB</small>
                        </label>
                    </article>
                    <div class="reference-results-column">
                        <article class="reference-card preview-card">
                            <div class="reference-card-title"><h2><span>♬</span>Audio Preview</h2><button class="format-chip" id="reference-clear">♜ &nbsp; Clear</button></div>
                            <div class="reference-wave-preview" aria-hidden="true"><div class="reference-wave-bars"></div></div>
                            <div class="reference-audio-controls"><button class="reference-play" id="reference-play" type="button" aria-label="Play audio">▶</button><span id="reference-current-time">00:00.00</span><input class="reference-seek" id="reference-seek" type="range" min="0" max="1000" value="0" aria-label="Seek audio" disabled><span id="reference-total-time">00:00.00</span></div>
                            <p class="reference-file-status" id="reference-file-status">No audio selected yet — upload a file to begin.</p>
                        </article>
                        <article class="reference-card result-card">
                            <div class="reference-card-title"><h2><span>♬</span>Analysis Result</h2><span class="result-spark">✦</span></div>
                            <div class="reference-result-content"><div class="reference-confidence-ring"><span>—</span></div><div class="reference-result-copy"><strong id="reference-result-label">Ready to analyze</strong><span class="reference-result-badge">Awaiting audio</span><p id="reference-result-summary">Your classification and confidence will appear here after a real scan.</p></div><div class="reference-bot" aria-hidden="true"><span>••</span></div></div>
                        </article>
                    </div>
                </section>
                <section class="reference-feature-row">
                    <button class="reference-feature-card" data-route="${APP_ROUTES.report}"><span class="feature-3d">▥</span><strong>Detailed Analysis<small>Comprehensive audio feature<br>analysis and explanations</small></strong><i>↗</i></button>
                    <button class="reference-feature-card" data-route="${APP_ROUTES.report}"><span class="feature-3d">▤</span><strong>Forensic Reports<small>Generate PDF reports for<br>detailed documentation</small></strong><i>↗</i></button>
                    <button class="reference-feature-card" data-route="${APP_ROUTES.repository}"><span class="feature-3d">◷</span><strong>Scan History<small>Keep track of your<br>analysis results</small></strong><i>↗</i></button>
                    <button class="reference-feature-card" data-route="${APP_ROUTES.diagnostics}"><span class="feature-3d">⚙</span><strong>System Diagnostics<small>Check model status<br>and system health</small></strong><i>↗</i></button>
                </section>
            </main>
        </section>
    `;
}

function workspaceScreen() {
    return `
        <section class="ui-screen screen-workspace">
            ${globalTopbar("workspace")}

            <main class="workspace-main">
                <div class="workspace-heading">
                    <div>
                        <div class="eyebrow">Laboratory Environment // Sector 08</div>
                        <h1>Detection Workspace</h1>
                    </div>
                    <div class="system-badge">
                        <span>System Status</span>
                        <strong id="workspace-system-status">Operational</strong>
                    </div>
                </div>

                <div class="workspace-grid">
                    <aside class="workspace-sidebar">
                        <div class="ingest-card">
                            <div class="card-head">
                                <span>⇄ Ingest New Media</span>
                                <span class="upload-cloud">☁</span>
                            </div>
                            <label id="dropzone" class="dropzone" for="audio-input">
                                <div class="dropzone-icon">⇪</div>
                                <strong>Drop Audio Source</strong>
                                <span>WAV, MP3, FLAC (MAX 10MB)</span>
                            </label>
                        </div>

                        <div class="param-card">
                            <h3>Analysis Parameters</h3>
                            <label class="range-label">
                                <span>Sensitivity Level</span>
                                <strong id="sensitivity-value">85%</strong>
                            </label>
                            <input id="sensitivity-slider" class="range-input" type="range" min="0" max="100" value="85">
                            <div class="range-scale">
                                <span>Broad</span>
                                <span>Forensic Precision</span>
                            </div>

                            <label class="field-group">
                                <span>Metadata Deep-Scan</span>
                                <select id="deep-scan-select">
                                    <option>Full Binary Reconfiguration</option>
                                    <option>Signal Integrity Pass</option>
                                    <option>Rapid Triage Sweep</option>
                                </select>
                            </label>

                            <label class="toggle-row">
                                <div>
                                    <strong>Isolate Biometrics</strong>
                                    <span>Enable neural voice print matching</span>
                                </div>
                                <input id="biometric-toggle" type="checkbox">
                            </label>

                            <button id="run-detection-btn" class="btn btn-solid btn-block">Run Detection</button>
                        </div>
                    </aside>

                    <section class="workspace-stage">
                        <div class="exhibit-header">
                            <div>
                                <div class="chip-row">
                                    <span class="chip chip-filled">Active Exhibit</span>
                                    <span id="workspace-exhibit-id" class="muted">ID: —</span>
                                </div>
                                <h2 id="workspace-filename">No file loaded</h2>
                            </div>
                            <div class="toolbar-icons">
                                <button class="icon-btn" id="workspace-download">⇩</button>
                                <button class="icon-btn" id="workspace-share">⇄</button>
                                <button class="icon-btn danger" id="workspace-clear">⌫</button>
                            </div>
                        </div>

                        <div class="waveform-surface">
                            <canvas id="workspace-waveform"></canvas>
                        </div>

                        <div class="stat-grid">
                            <div class="stat-card"><span>Peak Amplitude</span><strong id="stat-peak">—</strong></div>
                            <div class="stat-card"><span>Floor Noise</span><strong id="stat-noise">—</strong></div>
                            <div class="stat-card"><span>Duration</span><strong id="stat-duration">—</strong></div>
                            <div class="stat-card"><span>Sample Rate</span><strong id="stat-samplerate">—</strong></div>
                        </div>

                        <div class="player-row">
                            <div class="transport">
                                <button class="transport-btn" id="seek-back">◀◀</button>
                                <button class="transport-btn primary" id="play-toggle">▶</button>
                                <button class="transport-btn" id="seek-forward">▶▶</button>
                            </div>
                            <div class="time-display">
                                <span id="current-time">00:00.00</span>
                                <span>/</span>
                                <span id="total-time">00:00.00</span>
                            </div>
                            <div class="volume-box">
                                <span>🔊</span>
                                <input id="volume-slider" type="range" min="0" max="1" step="0.01" value="0.8">
                            </div>
                        </div>

                        <div class="analysis-log">
                            <div class="log-head">Analysis Log: Core_Engine_v4.2</div>
                            <div id="workspace-log-body" class="log-body">
                                <div>[09:21:04] Initializing VoxTrust detection engine...</div>
                                <div>[09:21:05] Awaiting evidence intake from secure client.</div>
                                <div>[09:21:08] Parsing format headers and preparing waveform preview.</div>
                                <div>[09:21:10] Standing by for artifact scan.</div>
                            </div>
                        </div>
                    </section>
                </div>
            </main>
        </section>
    `;
}

function processingScreen() {
    return `
        <section class="ui-screen screen-processing">
            ${globalTopbar("processing")}

            <main class="processing-main">
                <aside class="processing-sidebar">
                    <div class="case-panel">
                        <div class="case-title" id="processing-case-title">CASE_082_ALPHA</div>
                        <div class="status-pill">Status: Processing</div>
                    </div>
                    <nav class="stack-nav">
                        <button class="stack-link active" data-report-tab="spectrogram">Spectrogram</button>
                        <button class="stack-link" data-route="${APP_ROUTES.repository}">Biometrics</button>
                        <button class="stack-link" data-report-tab="metadata">Metadata</button>
                        <button class="stack-link" data-report-tab="transcription">Transcription</button>
                        <button class="stack-link" data-route="${APP_ROUTES.report}">Report</button>
                    </nav>
                    <button class="btn btn-solid btn-block" id="export-evidence-btn">Export Evidence</button>
                    <button class="text-link left" data-route="${APP_ROUTES.diagnostics}">System Logs</button>
                </aside>

                <section class="processing-stage">
                    <div class="processing-header">
                        <div>
                            <div class="eyebrow">Current Active Operation</div>
                            <h1>Analyzing Voice Patterns...</h1>
                            <p>Subject: <span id="processing-subject" class="subject-name">'Unknown_Capture_021.wav'</span></p>
                        </div>
                        <div class="confidence-index">
                            <strong id="processing-confidence">—</strong>
                            <span>Current Confidence Index</span>
                        </div>
                    </div>

                    <div class="processing-layout">
                        <div class="processing-visual-card">
                            <canvas id="processing-waveform"></canvas>
                            <div class="temporal-sync"><span></span>Temporal Sync: Stable</div>
                        </div>
                        <div class="processing-sidecards">
                            <article class="meta-card">
                                <h3>Technical Metadata</h3>
                                <dl>
                                    <div><dt>Processing Engine</dt><dd>VOX-CORE v2.4.0</dd></div>
                                    <div><dt>Sample Rate</dt><dd id="processing-rate">—</dd></div>
                                    <div><dt>Transport Security</dt><dd>HTTPS in transit</dd></div>
                                    <div><dt>Vector Alignment</dt><dd>Optimized</dd></div>
                                </dl>
                            </article>
                            <article class="warning-card">
                                <h3>Cautionary Note</h3>
                                <p id="processing-warning">
                                    Analysis integrity depends on signal-to-noise ratio. Ambient noise floor detected in the evidence capture.
                                </p>
                            </article>
                        </div>
                    </div>

                    <div class="execution-card">
                        <h3>Execution Stack</h3>
                        <div id="processing-steps" class="execution-list">
                            <div class="exec-step complete" data-step="decode"><span>Decoding and validating audio...</span><em>Complete (0.3s)</em></div>
                            <div class="exec-step active" data-step="features"><span>Extracting features...</span><em>Processing...</em></div>
                            <div class="exec-step queued" data-step="classify"><span>Running classifier...</span><em>Queued</em></div>
                            <div class="exec-step queued" data-step="calibrate"><span>Calibrating confidence...</span><em>Queued</em></div>
                            <div class="exec-step queued" data-step="finalize"><span>Final synthesis...</span><em>Queued</em></div>
                        </div>
                    </div>
                </section>

                <aside class="processing-actions">
                    <button class="btn btn-danger-outline btn-block" id="cancel-analysis-btn">Cancel Analysis</button>
                </aside>
            </main>
        </section>
    `;
}

function reportScreen() {
    return `
        <section class="ui-screen screen-report">
            ${globalTopbar("report")}

            <main class="report-main">
                <aside class="report-sidebar">
                    <div class="analyst-box">
                        <div class="avatar-chip large">A</div>
                        <div>
                            <strong>ANALYST_01</strong>
                            <span>Verified status: active</span>
                        </div>
                    </div>
                    <nav class="report-nav">
                        <button data-route="${APP_ROUTES.workspace}">Dashboard</button>
                        <button class="active" data-report-tab="forensic-log">Forensic Log</button>
                        <button data-route="${APP_ROUTES.repository}">Voice Prints</button>
                        <button data-route="${APP_ROUTES.status}">Network Stats</button>
                        <button data-route="${APP_ROUTES.repository}">Archive</button>
                    </nav>
                    <button class="btn btn-solid btn-block muted-btn">Upgrade Protocol</button>
                    <div class="report-support">
                        <button class="text-link left" data-route="${APP_ROUTES.status}">Support</button>
                        <button class="text-link left" data-route="${APP_ROUTES.home}">Log Out</button>
                    </div>
                </aside>

                <section class="report-content">
                    <div class="report-head">
                        <div>
                            <div class="eyebrow" id="report-scan-id">SCAN_ID: —</div>
                            <h1>Forensic Analysis</h1>
                        </div>
                        <div class="report-actions">
                            <button class="btn btn-ghost" id="share-result-btn">Share Result</button>
                            <button class="btn btn-solid" id="download-report-btn">Download Full Report (PDF)</button>
                        </div>
                    </div>

                    <div class="report-layout">
                        <article class="verdict-card">
                            <div class="verdict-chip">VERDICT_CONFIRMED</div>
                            <h2 id="report-verdict">—</h2>
                            <p id="report-summary">Run a scan to see the forensic verdict.</p>
                            <div class="confidence-band">
                                <div>
                                    <span>Confidence Score</span>
                                    <div class="confidence-meter"><span id="report-confidence-bar"></span></div>
                                </div>
                                <strong id="report-confidence-label">—</strong>
                            </div>
                        </article>

                        <aside class="report-side-stack">
                            <article class="meta-card report-meta">
                                <h3>Technical Metadata</h3>
                                <dl>
                                    <div><dt>Container</dt><dd id="report-format">—</dd></div>
                                    <div><dt>Bitrate</dt><dd id="report-bitrate">—</dd></div>
                                    <div><dt>Sample Rate</dt><dd id="report-rate">—</dd></div>
                                    <div><dt>Channels</dt><dd id="report-channels">—</dd></div>
                                </dl>
                                <div class="fingerprint-box">
                                    <span>Forensic Fingerprint</span>
                                    <code id="report-fingerprint">—</code>
                                </div>
                            </article>
                            <article class="insight-card">
                                <h3>Model Insights</h3>
                                <div id="report-insights" class="insight-list"></div>
                            </article>
                        </aside>
                    </div>

                    <section class="audio-evidence-card">
                        <div class="section-title">Audio Evidence</div>
                        <canvas id="report-waveform"></canvas>
                        <div class="player-row compact">
                            <div class="transport">
                                <button class="transport-btn" id="report-seek-back">◀◀</button>
                                <button class="transport-btn primary" id="report-play-toggle">▶</button>
                                <button class="transport-btn" id="report-seek-forward">▶▶</button>
                            </div>
                            <div class="time-display">
                                <span id="report-current-time">00:00.00</span>
                                <span>/</span>
                                <span id="report-total-time">00:00.00</span>
                            </div>
                        </div>
                    </section>
                </section>
            </main>
        </section>
    `;
}

function diagnosticsScreen() {
    return `
        <section class="ui-screen screen-diagnostics">
            ${globalTopbar("diagnostics")}

            <main class="diagnostics-main">
                <section class="diagnostics-hero">
                    <div class="eyebrow">Incident Review</div>
                    <h1>Diagnostic Report</h1>
                    <p id="diagnostics-summary">
                        The forensic integrity engine encountered technical friction during the last analysis cycle.
                    </p>
                </section>

                <section class="diagnostics-grid">
                    <article id="diag-format" class="diagnostic-card">
                        <div class="diagnostic-icon">⌁</div>
                        <h2>Invalid File Format</h2>
                        <p id="diag-format-text">
                            Technical validation failed. The uploaded container is not supported by the spectral analysis engine.
                        </p>
                        <div class="diagnostic-actions">
                            <button class="btn btn-ghost" data-route="${APP_ROUTES.workspace}">Try Again</button>
                            <button class="btn btn-ghost" data-route="${APP_ROUTES.status}">View Supported Formats</button>
                        </div>
                    </article>

                    <article id="diag-sample" class="diagnostic-card">
                        <div class="diagnostic-icon">≈</div>
                        <h2>Insufficient Sample</h2>
                        <p id="diag-sample-text">
                            Biometric fingerprinting requires a minimum threshold of audio evidence.
                        </p>
                        <div class="diagnostic-actions">
                            <button class="btn btn-ghost" data-route="${APP_ROUTES.workspace}">Upload Longer Sample</button>
                        </div>
                    </article>

                    <article id="diag-connection" class="diagnostic-card">
                        <div class="diagnostic-icon">⌿</div>
                        <h2>Connection Lost</h2>
                        <p id="diag-connection-text">
                            The secure tunnel to the analysis cluster was unexpectedly severed.
                        </p>
                        <div class="diagnostic-actions">
                            <button class="btn btn-ghost" id="retry-analysis-btn">Retry Uplink</button>
                        </div>
                    </article>
                </section>

                <section class="diagnostics-foot">
                    <article class="protocol-card">
                        <h3>Forensic Protocols</h3>
                        <ul>
                            <li>All diagnostic errors are logged in accordance with the VOX-STD-9912 compliance framework.</li>
                            <li>Chain of custody for audio samples remains intact despite analysis interruption.</li>
                            <li>Results generated after resolution maintain full legal admissibility.</li>
                        </ul>
                    </article>
                    <article class="system-log-card">
                        <h3>System Status Log</h3>
                        <pre id="diagnostics-log">SESSION_ID: TX-8829-001X
LATENCY: TIMEOUT
KERNEL: VOXTRUST-OS-V4.2
2024-10-24T14:22:01Z ERR_CONNECTION_REFUSED</pre>
                        <div class="diagnostics-visual"></div>
                    </article>
                </section>
            </main>
        </section>
    `;
}

function repositoryScreen() {
    return `
        <section class="ui-screen screen-repository">
            ${globalTopbar("repository")}

            <main class="repository-main">
                <div class="eyebrow">History Of Analysis</div>
                <h1>Forensic Archive</h1>

                <div class="filter-row">
                    <label class="field-group grow">
                        <span>Search filename</span>
                        <input id="archive-search" type="search" placeholder="Query database...">
                    </label>
                    <label class="field-group">
                        <span>Classification</span>
                        <select id="archive-classification">
                            <option value="ALL">All Assets</option>
                            <option value="HUMAN">Human</option>
                            <option value="AI_GENERATED">AI Generated</option>
                            <option value="INCONCLUSIVE">Inconclusive</option>
                        </select>
                    </label>
                    <label class="field-group">
                        <span>Temporal Range</span>
                        <input id="archive-date" type="date">
                    </label>
                    <button id="apply-filters-btn" class="btn btn-solid">Apply Filters</button>
                </div>

                <div class="archive-table-card">
                    <div class="archive-header-row">
                        <span>Signature Waveform</span>
                        <span>Asset Identifier</span>
                        <span>Timestamp (UTC)</span>
                        <span>Classification</span>
                        <span>Confidence</span>
                        <span>Actions</span>
                    </div>
                    <div id="archive-rows" class="archive-rows"></div>
                </div>

                <div class="pagination-row">
                    <span id="archive-summary">Showing 0 of 0 records</span>
                    <div class="page-actions">
                        <button id="archive-prev" class="btn btn-ghost">Previous</button>
                        <button id="archive-next" class="btn btn-solid">Next</button>
                    </div>
                </div>
            </main>
        </section>
    `;
}

function statusScreen() {
    return `
        <section class="ui-screen screen-status">
            ${globalTopbar("status")}

            <main class="status-main">
                <section class="status-hero">
                    <div>
                        <h1>System Status</h1>
                        <p>Technical diagnostics and operational methodology documentation.</p>
                    </div>
                    <span id="status-version-chip" class="chip chip-filled">v2.4.0 Stable</span>
                </section>

                <div class="status-grid">
                    <section class="status-panel">
                        <div class="panel-head">
                            <h2>Core Engines</h2>
                            <span id="status-summary-pill" class="status-dot">All systems functional</span>
                        </div>
                        <div id="engine-status-list" class="engine-status-list"></div>
                    </section>

                    <section class="node-panel">
                        <h2>Infrastructure Nodes</h2>
                        <div class="node-board">
                            <div class="node-chip">NODE_WEST_01</div>
                            <div class="node-chip">NODE_EUR_04</div>
                            <div class="node-chip">NODE_ASIA_02</div>
                            <div class="circuit-panel">
                                <div class="mesh-card">
                                    <strong>Global Mesh Active</strong>
                                    <p>Decentralized forensic nodes ensuring low-latency processing and redundant verification chains.</p>
                                </div>
                            </div>
                            <div class="node-foot">
                                <span>Status: Encrypted Tunnel</span>
                                <span id="node-ping">92ms Ping</span>
                            </div>
                        </div>
                    </section>
                </div>

                <div class="status-grid lower">
                    <section class="spec-panel">
                        <h2>Forensic Specifications</h2>
                        <div class="spec-grid">
                            <article>
                                <span>Max File Capacity</span>
                                <strong id="status-max-size">10.0 MB</strong>
                                <p>Strict limitation enforced for volatile memory buffer stability during scanning.</p>
                            </article>
                            <article>
                                <span>Supported Encoding</span>
                                <strong id="status-supported-formats">MP3 / WAV / FLAC</strong>
                                <p>Lossless and compressed forensic formats supported by the live API.</p>
                            </article>
                            <article>
                                <span>Processing Latency</span>
                                <strong id="status-latency">1.2 sec/min</strong>
                                <p>Average extraction time per minute of audio stream analyzed.</p>
                            </article>
                            <article>
                                <span>Active Dialects</span>
                                <strong id="status-languages">EN, HI, TA, +2</strong>
                                <p>Automatic language switching enabled for supported multi-speaker dialogs.</p>
                            </article>
                        </div>
                    </section>

                    <section class="privacy-panel">
                        <h2>Privacy Protocol</h2>
                        <article>
                            <h3>Zero-Retention Architecture</h3>
                            <p>
                                Voice Forensics does not store uploaded audio files on persistent storage. Analysis is performed in isolated volatile memory and purged after session termination.
                            </p>
                        </article>
                        <article>
                            <h3>Anonymized Metadata</h3>
                            <p>
                                Resultant technical metadata is stripped of personally identifying context before it is surfaced to operator dashboards.
                            </p>
                        </article>
                    </section>
                </div>
            </main>
        </section>
    `;
}
