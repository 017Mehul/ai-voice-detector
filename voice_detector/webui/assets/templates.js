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
        <section class="ui-screen screen-home">
            ${globalTopbar("home")}

            <main class="home-main">
                <section class="hero-panel">
                    <div class="hero-copy">
                        <div class="eyebrow">Forensic Authority v4.8</div>
                        <h1>IS IT HUMAN<br>OR AI?</h1>
                        <p>
                            Detect synthetic speech and deepfake audio with laboratory-grade precision.
                            VoxTrust provides the definitive verdict for legal, journalistic, and security investigations.
                        </p>
                        <div class="hero-actions">
                            <button class="btn btn-solid" data-route="${APP_ROUTES.workspace}">Analyze Audio</button>
                            <button class="btn btn-ghost" data-route="${APP_ROUTES.status}">View Demo</button>
                        </div>
                    </div>
                    <div class="hero-visual">
                        <div class="wave-card">
                            <div class="wave-card-head">
                                <span>Forensic Pattern Map</span>
                                <span>D-4582</span>
                            </div>
                            <div class="hero-wave-grid">
                                <div class="hero-bars hero-bars-gold"></div>
                                <div class="hero-bars hero-bars-cyan"></div>
                            </div>
                        </div>
                        <div class="confidence-card">
                            <span>Neural Confidence</span>
                            <strong>99.8%</strong>
                            <small>Verified Authentic</small>
                        </div>
                    </div>
                </section>

                <section class="compatibility-strip">
                    <div class="section-intro">
                        <h2>Engine Compatibility</h2>
                        <p>Our neural engine supports lossless and compressed formats for exhaustive frequency analysis.</p>
                    </div>
                    <div class="format-grid">
                        <div class="format-card"><strong>MP3</strong><span>MPEG Layer III</span></div>
                        <div class="format-card"><strong>WAV</strong><span>Waveform Audio</span></div>
                        <div class="format-card"><strong>FLAC</strong><span>Free Lossless</span></div>
                    </div>
                </section>

                <section class="feature-grid">
                    <article class="feature-card">
                        <div class="feature-icon">◌</div>
                        <h3>Forensic Anonymity</h3>
                        <p>We prioritize institutional security. All analysis is performed within a sandboxed environment.</p>
                        <ul>
                            <li>AES-256 encryption at rest</li>
                            <li>Zero-retention processing</li>
                        </ul>
                    </article>
                    <article class="feature-card">
                        <div class="feature-icon">◎</div>
                        <h3>Multi-Lingual Engine</h3>
                        <p>Voice signatures transcend language. Our neural network analyzes phonetic artifacts across supported dialects.</p>
                        <ul>
                            <li>Phonetic artifact mapping</li>
                            <li>Global dialect support</li>
                        </ul>
                    </article>
                </section>

                <section class="process-strip">
                    <div class="eyebrow accent">Protocol</div>
                    <h2>The Process</h2>
                    <div class="process-grid">
                        <article class="process-card process-card-a">
                            <span>01</span>
                            <h3>Upload Source</h3>
                            <p>Import raw audio through our secure interface. We accept MP3, WAV, and FLAC evidence.</p>
                        </article>
                        <article class="process-card process-card-b">
                            <span>02</span>
                            <h3>Neural Analysis</h3>
                            <p>Our engine scans for noise-floor drift, pitch-perfect harmonics, and synthetic reverberation patterns.</p>
                        </article>
                        <article class="process-card process-card-c">
                            <span>03</span>
                            <h3>Verdict Report</h3>
                            <p>Receive a comprehensive forensic report detailing synthetic risk, signal artifacts, and evidence notes.</p>
                        </article>
                    </div>
                </section>

                <section class="case-study">
                    <div class="case-image">
                        <div class="eye-plate"></div>
                        <div class="quote-card">
                            <small>Case ID: 77-X</small>
                            <p>"The impersonation was perfect to the human ear, but VoxTrust saw the mathematical ghost in the waveform."</p>
                        </div>
                    </div>
                    <div class="case-copy">
                        <div class="eyebrow">Investigation</div>
                        <h2>The Scandinavian CEO Heist</h2>
                        <p>
                            In March 2024, a major financial institution was targeted by a sophisticated voice-cloning attack.
                            VoxTrust isolated synthetic markers in the intercepted call and flagged it before the transfer was authorized.
                        </p>
                        <div class="case-score">99.98% Synthetic</div>
                        <button class="text-link" data-route="${APP_ROUTES.report}">Read The Full Investigation</button>
                    </div>
                </section>

                <section class="cta-band">
                    <h2>Secure Your Audio Integrity</h2>
                    <p>Join 400+ forensic labs and financial institutions protecting the truth in the age of AI.</p>
                    <div class="cta-actions">
                        <button class="btn btn-ghost-light" data-route="${APP_ROUTES.status}">Create Account</button>
                        <button class="btn btn-outline-light" data-route="${APP_ROUTES.workspace}">Talk To An Expert</button>
                    </div>
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
                            <strong id="processing-confidence">74.2%</strong>
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
                                    <div><dt>Sample Rate</dt><dd id="processing-rate">48.0 kHz / 24-bit</dd></div>
                                    <div><dt>Security Protocol</dt><dd>AES-256 Encrypted</dd></div>
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
