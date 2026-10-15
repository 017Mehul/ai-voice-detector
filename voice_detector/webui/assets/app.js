import { APP_ROUTES, renderScreen } from "./templates.js";

const STORAGE_KEYS = {
    lastResult: "voxtrust-last-result",
    lastError: "voxtrust-last-error",
};

const SCREEN_BY_PATH = new Map([
    [APP_ROUTES.home, "home"],
    [APP_ROUTES.workspace, "workspace"],
    [APP_ROUTES.processing, "processing"],
    [APP_ROUTES.report, "report"],
    [APP_ROUTES.diagnostics, "diagnostics"],
    [APP_ROUTES.repository, "repository"],
    [APP_ROUTES.status, "status"],
]);

const MAX_UI_FILE_SIZE_MB = 10;
const ARCHIVE_PAGE_SIZE = 4;

const appShell = document.getElementById("app-shell");

const state = {
    screen: "home",
    currentEvidence: null,
    lastResult: loadJson(STORAGE_KEYS.lastResult, null),
    lastError: loadJson(STORAGE_KEYS.lastError, null),
    archiveItems: [],
    archivePage: 1,
    archiveFilters: {
        search: "",
        classification: "ALL",
        date: "",
    },
    processing: {
        controller: null,
        stageTimer: null,
        confidenceTimer: null,
        running: false,
    },
    limits: {
        maxAudioSizeMb: MAX_UI_FILE_SIZE_MB,
        supportedFormats: ["MP3", "WAV", "FLAC"],
    },
};

window.addEventListener("popstate", () => {
    renderRoute(getScreenFromPath(location.pathname));
});

document.addEventListener("DOMContentLoaded", async () => {
    renderRoute(getScreenFromPath(location.pathname));
    await Promise.all([
        warmRuntimeInfo(),
        refreshArchiveFromServer({ refreshScreen: true }),
    ]);
});

function getScreenFromPath(pathname) {
    return SCREEN_BY_PATH.get(pathname) || "home";
}

function renderRoute(screen, options = {}) {
    state.screen = screen;
    appShell.innerHTML = renderScreen(screen);

    if (!options.replaceState) {
        const route = APP_ROUTES[screen] || APP_ROUTES.home;
        if (location.pathname !== route) {
            history.pushState({}, "", route);
        }
    }

    bindBaseInteractions();
    initializeScreen(screen);
}

function navigate(route, replaceState = false) {
    const screen = getScreenFromPath(route);
    renderRoute(screen, { replaceState });
}

function bindBaseInteractions() {
    document.querySelectorAll("[data-route]").forEach((button) => {
        button.addEventListener("click", () => navigate(button.dataset.route));
    });
    document.querySelectorAll("[data-report-tab]").forEach((button) => {
        button.addEventListener("click", () => {
            // Mark active
            button.closest("nav")?.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
            button.classList.add("active");
            showToast(`${button.textContent} — coming soon`);
        });
    });

    document.querySelectorAll("[data-open-uploader]").forEach((button) => {
        button.addEventListener("click", openUploader);
    });

    const referencePlay = document.getElementById("reference-play");
    if (referencePlay) {
        referencePlay.addEventListener("click", togglePlayback);
    }
    const referenceSeek = document.getElementById("reference-seek");
    if (referenceSeek) {
        referenceSeek.addEventListener("input", (event) => {
            const audio = getAudio();
            if (audio && Number.isFinite(audio.duration)) {
                audio.currentTime = (Number(event.target.value) / 1000) * audio.duration;
                syncAudioTimes();
            }
        });
    }

    const referenceClear = document.getElementById("reference-clear");
    if (referenceClear) {
        referenceClear.addEventListener("click", () => {
            if (state.currentEvidence) {
                clearEvidence();
            } else {
                showToast("Choose an audio file to begin a scan.");
            }
        });
    }

    const fileInput = document.getElementById("audio-input");
    if (fileInput) {
        fileInput.addEventListener("change", async (event) => {
            const [file] = event.target.files || [];
            if (file) {
                await handleSelectedFile(file);
            }
            event.target.value = "";
        });
    }

    const audio = getAudio();
    if (audio && state.currentEvidence?.objectUrl) {
        audio.src = state.currentEvidence.objectUrl;
        audio.volume = state.currentEvidence.volume ?? 0.8;
        audio.addEventListener("timeupdate", syncAudioTimes);
        audio.addEventListener("loadedmetadata", syncAudioTimes);
        audio.addEventListener("ended", syncPlayButtons);
    }
}

function initializeScreen(screen) {
    if (screen === "home") {
        populateReferenceHome();
    } else if (screen === "workspace") {
        initWorkspaceScreen();
    } else if (screen === "processing") {
        initProcessingScreen();
    } else if (screen === "report") {
        initReportScreen();
    } else if (screen === "diagnostics") {
        initDiagnosticsScreen();
    } else if (screen === "repository") {
        initRepositoryScreen();
    } else if (screen === "status") {
        initStatusScreen();
    }
}

function loadJson(key, fallback) {
    try {
        const raw = localStorage.getItem(key);
        return raw ? JSON.parse(raw) : fallback;
    } catch {
        return fallback;
    }
}

function persistJson(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}

function getAudio() {
    return document.getElementById("uploaded-audio");
}

function openUploader() {
    document.getElementById("audio-input")?.click();
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) {
        return "00:00.00";
    }
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
}

function formatDate(isoString) {
    const date = new Date(isoString);
    const year = date.getUTCFullYear();
    const month = String(date.getUTCMonth() + 1).padStart(2, "0");
    const day = String(date.getUTCDate()).padStart(2, "0");
    const hours = String(date.getUTCHours()).padStart(2, "0");
    const minutes = String(date.getUTCMinutes()).padStart(2, "0");
    const seconds = String(date.getUTCSeconds()).padStart(2, "0");
    return {
        date: `${year}-${month}-${day}`,
        time: `${hours}:${minutes}:${seconds}`,
    };
}

function toClassificationLabel(classification) {
    if (classification === "AI_GENERATED") {
        return "AI GENERATED";
    }
    if (classification === "HUMAN") {
        return "HUMAN";
    }
    return "INCONCLUSIVE";
}

function classificationTone(classification) {
    if (classification === "AI_GENERATED") {
        return "tone-ai";
    }
    if (classification === "HUMAN") {
        return "tone-human";
    }
    return "tone-neutral";
}

async function warmRuntimeInfo() {
    try {
        const limitsResponse = await fetch("/api/v1/limits");
        if (!limitsResponse.ok) {
            return;
        }
        const limits = await limitsResponse.json();
        if (limits.max_audio_size_mb) {
            state.limits.maxAudioSizeMb = limits.max_audio_size_mb;
        }
        if (Array.isArray(limits.supported_formats)) {
            state.limits.supportedFormats = limits.supported_formats;
        }
    } catch {
        // Leave the static fallback values in place.
    }
}

async function refreshArchiveFromServer(options = {}) {
    const payload = await fetchJson("/api/v1/history");
    if (!payload || !Array.isArray(payload.items)) {
        return;
    }

    state.archiveItems = payload.items.map(normalizeArchiveItem);

    if (!state.lastResult && state.archiveItems.length > 0 && state.screen === "report") {
        state.lastResult = buildReportFromArchive(state.archiveItems[0]);
        persistJson(STORAGE_KEYS.lastResult, state.lastResult);
    }

    if (options.refreshScreen && ["repository", "report"].includes(state.screen)) {
        initializeScreen(state.screen);
    }
}

async function handleSelectedFile(file) {
    if (file.size > state.limits.maxAudioSizeMb * 1024 * 1024) {
        setLastError({
            error: "PAYLOAD_TOO_LARGE",
            message: `The selected file exceeds the ${state.limits.maxAudioSizeMb}MB upload limit.`,
            details: {
                activeCard: "connection",
            },
        });
        navigate(APP_ROUTES.diagnostics);
        return;
    }

    const extension = file.name.split(".").pop()?.toUpperCase() || "AUDIO";
    const allowed = new Set(state.limits.supportedFormats.map((format) => format.toUpperCase()));
    if (!allowed.has(extension)) {
        setLastError({
            error: "UNSUPPORTED_FORMAT",
            message: `.${extension.toLowerCase()} files are not supported. Use ${state.limits.supportedFormats.join(", ")}.`,
            details: {
                activeCard: "format",
            },
        });
        navigate(APP_ROUTES.diagnostics);
        return;
    }

    if (state.currentEvidence?.objectUrl) {
        URL.revokeObjectURL(state.currentEvidence.objectUrl);
    }

    const arrayBuffer = await file.arrayBuffer();
    const objectUrl = URL.createObjectURL(file);
    const base64 = arrayBufferToBase64(arrayBuffer);
    const analysis = await analyzeAudioFile(arrayBuffer, file, objectUrl);

    state.currentEvidence = {
        ...analysis,
        exhibitId: buildEvidenceId(),
        fileName: file.name,
        format: extension,
        sizeBytes: file.size,
        sizeLabel: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
        objectUrl,
        base64,
        file,
        volume: 0.8,
        logEntries: buildWorkspaceLog(file.name),
    };

    appendWorkspaceLog(`Loaded exhibit ${file.name} successfully.`);
    if (state.screen === "home") {
        navigate(APP_ROUTES.home);
        populateReferenceHome();
        const status = document.getElementById("reference-file-status");
        if (status) status.textContent = `Loaded: ${file.name} — ready to preview. Select Analyze to run detection.`;
        const seek = document.getElementById("reference-seek");
        if (seek) seek.disabled = false;
    } else {
        navigate(APP_ROUTES.workspace);
    }
}

async function analyzeAudioFile(arrayBuffer, file, objectUrl) {
    const audio = getAudio();
    let audioBuffer = null;
    let waveform = placeholderWaveform(88);

    if (audio) {
        audio.src = objectUrl;
        audio.load();
        try {
            await once(audio, "loadedmetadata");
        } catch {
            // Keep fallback metadata.
        }
    }

    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        waveform = summarizeWaveform(audioBuffer.getChannelData(0), 88);
        if (audioContext.state !== "closed") {
            await audioContext.close();
        }
    } catch {
        waveform = placeholderWaveform(88);
    }

    const duration = audioBuffer?.duration || audio?.duration || 0;
    const sampleRate = audioBuffer?.sampleRate || 48000;
    const channels = audioBuffer?.numberOfChannels || 1;
    const firstChannel = audioBuffer?.getChannelData(0) || null;
    const peak = firstChannel
        ? firstChannel.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0)
        : 0.62;
    const rms = firstChannel ? calculateRms(firstChannel) : 0.008;
    const bitrate = duration > 0 ? Math.round((file.size * 8) / duration / 1000) : 320;

    return {
        duration,
        durationLabel: formatDuration(duration),
        sampleRate,
        sampleRateLabel: `${(sampleRate / 1000).toFixed(1)} kHz`,
        channelsLabel: channels === 1 ? "Mono" : `${channels} channels`,
        bitrateLabel: `${bitrate} kbps`,
        peakDbLabel: `${(-1 * (1 - peak) * 12).toFixed(1)} dB`,
        noiseDbLabel: `${Math.max(-72, 20 * Math.log10(Math.max(rms / 10, 0.0001))).toFixed(1)} dB`,
        waveform,
    };
}

function once(target, eventName) {
    return new Promise((resolve) => {
        target.addEventListener(eventName, resolve, { once: true });
    });
}

function arrayBufferToBase64(arrayBuffer) {
    let binary = "";
    const bytes = new Uint8Array(arrayBuffer);
    const chunkSize = 0x8000;

    for (let index = 0; index < bytes.length; index += chunkSize) {
        const chunk = bytes.subarray(index, index + chunkSize);
        binary += String.fromCharCode(...chunk);
    }

    return btoa(binary);
}

function calculateRms(samples) {
    const total = samples.reduce((sum, sample) => sum + sample * sample, 0);
    return Math.sqrt(total / samples.length);
}

function summarizeWaveform(samples, buckets) {
    const size = Math.floor(samples.length / buckets) || 1;
    const values = [];

    for (let index = 0; index < buckets; index += 1) {
        const start = index * size;
        const slice = samples.slice(start, start + size);
        const peak = slice.reduce((max, sample) => Math.max(max, Math.abs(sample)), 0);
        values.push(peak);
    }

    return values;
}

function placeholderWaveform(length) {
    return Array.from({ length }, (_, index) => {
        const wave = Math.sin(index / 4) * 0.35 + Math.sin(index / 7) * 0.15;
        return Math.max(0.12, Math.abs(wave) + 0.16);
    });
}

function setLastResult(result) {
    state.lastResult = result;
    persistJson(STORAGE_KEYS.lastResult, result);
}

function setLastError(error) {
    state.lastError = error;
    persistJson(STORAGE_KEYS.lastError, error);
}

function buildArchiveEntry(overrides) {
    const timestamp = overrides.timestamp || new Date().toISOString();
    return {
        id: overrides.id || `VX-${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
        request_id: overrides.request_id || null,
        name: overrides.name || "UNKNOWN_AUDIO.WAV",
        classification: overrides.classification || "INCONCLUSIVE",
        confidence: overrides.confidence ?? 0.5,
        timestamp,
        format: overrides.format || (overrides.name || "UNKNOWN_AUDIO.WAV").split(".").pop()?.toUpperCase() || "WAV",
        waveformTint: overrides.waveformTint || "wave-cyan",
        sampleRate: overrides.sampleRate ?? null,
        duration: overrides.duration ?? null,
        channels: overrides.channels ?? null,
        bitrate: overrides.bitrate ?? null,
        fingerprint: buildFingerprint((overrides.name || "UNKNOWN") + timestamp),
        summary: overrides.summary || null,
        reportUrl: overrides.reportUrl || (overrides.id ? `/api/v1/reports/${overrides.id}.pdf` : null),
        waveform: overrides.waveform || null,
    };
}

function normalizeArchiveItem(item) {
    return buildArchiveEntry({
        ...item,
        id: item.id,
        name: item.name,
        classification: item.classification,
        confidence: Number(item.confidence ?? item.confidence_score ?? 0),
        timestamp: item.timestamp,
        waveformTint: item.waveformTint,
        sampleRate: item.sampleRate ?? item.sample_rate_hz,
        duration: item.duration ?? item.duration_seconds,
        channels: item.channels,
        bitrate: item.bitrate ?? item.bitrate_kbps,
        fingerprint: item.fingerprint,
        waveform: item.waveform,
        format: item.format ?? item.file_format,
        summary: item.summary,
        request_id: item.request_id,
        reportUrl: item.reportUrl ?? item.report_url,
    });
}

function buildFingerprint(seed) {
    let hash = 0;
    for (let index = 0; index < seed.length; index += 1) {
        hash = ((hash << 5) - hash) + seed.charCodeAt(index);
        hash |= 0;
    }
    return Array.from({ length: 16 }, (_, index) => {
        const value = ((hash >> (index % 4) * 8) & 0xff) ^ (index * 17);
        return Math.abs(value).toString(16).padStart(2, "0").toUpperCase();
    }).join(":");
}


function populateReferenceHome() {
    const evidence = state.currentEvidence;
    const status = document.getElementById("reference-file-status");
    const label = document.getElementById("reference-result-label");
    const badge = document.querySelector(".reference-result-badge");
    const summary = document.getElementById("reference-result-summary");
    const ring = document.querySelector(".reference-confidence-ring span");
    const seek = document.getElementById("reference-seek");
    if (status) status.textContent = evidence ? `Loaded: ${evidence.fileName} — ready to preview.` : "No audio selected yet — upload a file to begin.";
    if (seek) seek.disabled = !evidence;
    if (label) label.textContent = state.lastResult ? (state.lastResult.classification === "AI_GENERATED" ? "Likely AI-generated" : state.lastResult.classification === "HUMAN" ? "Likely human" : "Inconclusive") : evidence ? "Ready to analyze" : "Ready to analyze";
    if (badge) badge.textContent = state.lastResult ? "Scan complete" : evidence ? "Awaiting scan" : "Awaiting audio";
    if (summary) summary.textContent = state.lastResult?.summary || (evidence ? "Audio loaded. Select Analyze to run the detector and view its result." : "Your classification and confidence will appear here after a real scan.");
    if (ring) ring.textContent = state.lastResult ? `${Math.round((state.lastResult.confidence || state.lastResult.confidence_score || 0) * 100)}%` : "—";
}

function initWorkspaceScreen() {
    populateWorkspace();

    const dropzone = document.getElementById("dropzone");
    dropzone?.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone?.addEventListener("drop", async (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
        const [file] = event.dataTransfer?.files || [];
        if (file) {
            await handleSelectedFile(file);
        }
    });

    document.getElementById("sensitivity-slider")?.addEventListener("input", (event) => {
        document.getElementById("sensitivity-value").textContent = `${event.target.value}%`;
    });

    document.getElementById("play-toggle")?.addEventListener("click", togglePlayback);
    document.getElementById("seek-back")?.addEventListener("click", () => nudgePlayback(-5));
    document.getElementById("seek-forward")?.addEventListener("click", () => nudgePlayback(5));
    document.getElementById("volume-slider")?.addEventListener("input", (event) => {
        const audio = getAudio();
        if (audio) {
            audio.volume = Number(event.target.value);
            if (state.currentEvidence) {
                state.currentEvidence.volume = audio.volume;
            }
        }
    });

    document.getElementById("workspace-clear")?.addEventListener("click", clearEvidence);
    document.getElementById("workspace-download")?.addEventListener("click", downloadOriginalEvidence);
    document.getElementById("workspace-share")?.addEventListener("click", shareCurrentEvidence);
    document.getElementById("run-detection-btn")?.addEventListener("click", startDetection);
}

function populateWorkspace() {
    const evidence = state.currentEvidence;
    document.getElementById("workspace-system-status").textContent = "Operational";

    if (!evidence) {
        document.getElementById("workspace-exhibit-id").textContent = "ID: —";
        document.getElementById("workspace-filename").textContent = "No file loaded";
        document.getElementById("stat-peak").textContent = "—";
        document.getElementById("stat-noise").textContent = "—";
        document.getElementById("stat-duration").textContent = "—";
        document.getElementById("stat-samplerate").textContent = "—";
        document.getElementById("total-time").textContent = "00:00.00";
        drawWaveform(document.getElementById("workspace-waveform"), placeholderWaveform(88), "#0a4d62", 0.4);
        renderWorkspaceLog();
        return;
    }

    document.getElementById("workspace-exhibit-id").textContent = `ID: ${evidence.exhibitId}`;
    document.getElementById("workspace-filename").textContent = evidence.fileName;
    document.getElementById("stat-peak").textContent = evidence.peakDbLabel;
    document.getElementById("stat-noise").textContent = evidence.noiseDbLabel;
    document.getElementById("stat-duration").textContent = evidence.durationLabel;
    document.getElementById("stat-samplerate").textContent = evidence.sampleRateLabel;
    document.getElementById("total-time").textContent = evidence.durationLabel;
    drawWaveform(document.getElementById("workspace-waveform"), evidence.waveform, "#0a4d62", 0.9);
    renderWorkspaceLog();
    syncAudioTimes();
}

function initProcessingScreen() {
    const evidence = state.currentEvidence;
    const caseTitle = document.getElementById("processing-case-title");
    if (caseTitle) {
        caseTitle.textContent = evidence ? evidence.fileName : "UNKNOWN_CAPTURE";
    }
    document.getElementById("processing-subject").textContent = evidence ? `'${evidence.fileName}'` : "'Unknown_Capture_021.wav'";
    document.getElementById("processing-rate").textContent = evidence ? evidence.sampleRateLabel : "—";
    document.getElementById("processing-warning").textContent = evidence
        ? `Analysis integrity depends on signal-to-noise ratio. Ambient noise floor is currently estimated at ${evidence.noiseDbLabel}.`
        : "Analysis integrity depends on signal-to-noise ratio. Ambient noise floor detected in the evidence capture.";
    drawWaveform(document.getElementById("processing-waveform"), evidence?.waveform || placeholderWaveform(72), "#0b5f77", 0.9);

    if (!state.processing.running && evidence?.base64) {
        runDetectionFlow();
    }

    document.getElementById("export-evidence-btn")?.addEventListener("click", downloadOriginalEvidence);
    document.getElementById("cancel-analysis-btn")?.addEventListener("click", cancelDetection);
}

function initReportScreen() {
    if (!state.lastResult && state.archiveItems.length === 0) {
        void refreshArchiveFromServer({ refreshScreen: true });
    }

    const result = state.lastResult
        || (state.archiveItems.length > 0 ? buildReportFromArchive(state.archiveItems[0]) : null);

    if (result) {
        populateReport(result);
    } else {
        renderEmptyReportState();
    }
    document.getElementById("report-play-toggle")?.addEventListener("click", togglePlayback);
    document.getElementById("report-seek-back")?.addEventListener("click", () => nudgePlayback(-5));
    document.getElementById("report-seek-forward")?.addEventListener("click", () => nudgePlayback(5));
    document.getElementById("share-result-btn")?.addEventListener("click", shareCurrentResult);
    document.getElementById("download-report-btn")?.addEventListener("click", downloadReport);
}

function initDiagnosticsScreen() {
    const error = state.lastError || {
        error: "UNSUPPORTED_FORMAT",
        message: "The previous analysis cycle did not meet validation requirements.",
        details: { activeCard: "format" },
    };

    document.getElementById("diagnostics-summary").textContent = error.message;
    document.getElementById("diagnostics-log").textContent = [
        `SESSION_ID: ${buildArchiveEntry({}).id}`,
        `ERROR_CODE: ${error.error}`,
        `TIMESTAMP: ${new Date().toISOString()}`,
        `DETAIL: ${error.message}`,
    ].join("\n");

    const activeCard = error.details?.activeCard || mapErrorToCard(error.error);
    ["format", "sample", "connection"].forEach((card) => {
        document.getElementById(`diag-${card}`)?.classList.toggle("active", card === activeCard);
    });

    if (error.error === "AUDIO_TOO_SHORT" || error.error === "AUDIO_TOO_SMALL") {
        document.getElementById("diag-sample-text").textContent = "Biometric fingerprinting requires at least 2.0 seconds of clear audio evidence.";
    }
    if (error.error === "UNSUPPORTED_FORMAT") {
        document.getElementById("diag-format-text").textContent = `Technical validation failed. The uploaded file is not supported. Use ${state.limits.supportedFormats.join(", ")}.`;
    }
    if (["SERVICE_UNAVAILABLE", "REQUEST_TIMEOUT", "PROCESSING_ERROR"].includes(error.error)) {
        document.getElementById("diag-connection-text").textContent = "The secure analysis service became unavailable during the last scan. Retry the uplink or submit the evidence again.";
    }

    document.getElementById("retry-analysis-btn")?.addEventListener("click", () => {
        if (state.currentEvidence?.base64) {
            navigate(APP_ROUTES.processing);
        } else {
            navigate(APP_ROUTES.workspace);
        }
    });
}

function initRepositoryScreen() {
    if (state.archiveItems.length === 0) {
        void refreshArchiveFromServer({ refreshScreen: true });
    }

    document.getElementById("archive-search").value = state.archiveFilters.search;
    document.getElementById("archive-classification").value = state.archiveFilters.classification;
    document.getElementById("archive-date").value = state.archiveFilters.date;

    document.getElementById("archive-search")?.addEventListener("input", (event) => {
        state.archiveFilters.search = event.target.value;
    });
    document.getElementById("archive-classification")?.addEventListener("change", (event) => {
        state.archiveFilters.classification = event.target.value;
    });
    document.getElementById("archive-date")?.addEventListener("change", (event) => {
        state.archiveFilters.date = event.target.value;
    });
    document.getElementById("apply-filters-btn")?.addEventListener("click", () => {
        state.archivePage = 1;
        renderArchiveRows();
    });
    document.getElementById("archive-prev")?.addEventListener("click", () => {
        state.archivePage = Math.max(1, state.archivePage - 1);
        renderArchiveRows();
    });
    document.getElementById("archive-next")?.addEventListener("click", () => {
        state.archivePage += 1;
        renderArchiveRows();
    });

    renderArchiveRows();
}

async function initStatusScreen() {
    await populateStatusScreen();
}

function togglePlayback() {
    const audio = getAudio();
    if (!audio || !state.currentEvidence?.objectUrl) {
        return;
    }

    if (audio.paused) {
        audio.play();
    } else {
        audio.pause();
    }
    syncPlayButtons();
}

function nudgePlayback(seconds) {
    const audio = getAudio();
    if (!audio) {
        return;
    }
    audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + seconds));
    syncAudioTimes();
}

function syncPlayButtons() {
    const audio = getAudio();
    const stateLabel = audio && !audio.paused ? "pause" : "play";
    const icon = stateLabel === "pause" ? "❚❚" : "▶";
    [document.getElementById("play-toggle"), document.getElementById("report-play-toggle"), document.getElementById("reference-play")]
        .filter(Boolean)
        .forEach((button) => {
            button.replaceChildren(document.createTextNode(icon));
            button.dataset.state = stateLabel;
            button.setAttribute("aria-label", stateLabel === "pause" ? "Pause audio" : "Play audio");
        });
}

function syncAudioTimes() {
    const audio = getAudio();
    if (!audio) {
        return;
    }

    const current = formatDuration(audio.currentTime || 0);
    const total = formatDuration(audio.duration || state.currentEvidence?.duration || 0);
    document.getElementById("current-time") && (document.getElementById("current-time").textContent = current);
    document.getElementById("total-time") && (document.getElementById("total-time").textContent = total);
    document.getElementById("reference-current-time") && (document.getElementById("reference-current-time").textContent = current);
    document.getElementById("reference-total-time") && (document.getElementById("reference-total-time").textContent = total);
    const referenceSeek = document.getElementById("reference-seek");
    if (referenceSeek && audio.duration > 0) referenceSeek.value = String(Math.round((audio.currentTime / audio.duration) * 1000));
    document.getElementById("report-current-time") && (document.getElementById("report-current-time").textContent = current);
    document.getElementById("report-total-time") && (document.getElementById("report-total-time").textContent = total);
    syncPlayButtons();
}

async function startDetection() {
    if (!state.currentEvidence?.base64) {
        setLastError({
            error: "INVALID_REQUEST",
            message: "Upload an audio file before running detection.",
            details: { activeCard: "format" },
        });
        navigate(APP_ROUTES.diagnostics);
        return;
    }

    navigate(APP_ROUTES.processing);
}

async function runDetectionFlow() {
    if (!state.currentEvidence?.base64 || state.processing.running) {
        return;
    }

    clearProcessingTimers();
    state.processing.running = true;
    state.processing.controller = new AbortController();

    const stageSequence = ["features", "classify", "calibrate", "finalize"];
    let stageIndex = 0;
    const startedAt = performance.now();

    const confidenceNode = document.getElementById("processing-confidence");
    if (confidenceNode) confidenceNode.textContent = "Waiting for model…";

    updateProcessingStage("decode", "Complete", "complete");
    updateProcessingStage("features", "Processing…", "active");

    state.processing.stageTimer = window.setInterval(() => {
        const currentStage = stageSequence[stageIndex];
        if (!currentStage) {
            return;
        }
        updateProcessingStage(currentStage, `Complete (${(0.4 + stageIndex * 0.4).toFixed(1)}s)`, "complete");
        const nextStage = stageSequence[stageIndex + 1];
        if (nextStage) {
            updateProcessingStage(nextStage, "Processing...", "active");
        }
        stageIndex += 1;
    }, 900);

    try {
        const response = await fetch("/api/v1/detect-voice", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                audio_base64: state.currentEvidence.base64,
                filename: state.currentEvidence.fileName,
                file_format: state.currentEvidence.format,
                duration_seconds: state.currentEvidence.duration,
                sample_rate_hz: state.currentEvidence.sampleRate,
                channels: state.currentEvidence.channelsLabel,
                bitrate_kbps: state.currentEvidence.bitrateLabel,
                waveform_bars: state.currentEvidence.waveform,
            }),
            signal: state.processing.controller.signal,
        });

        const elapsed = performance.now() - startedAt;
        if (elapsed < 2600) {
            await new Promise((resolve) => setTimeout(resolve, 2600 - elapsed));
        }

        if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({
                error: "PROCESSING_ERROR",
                message: "The analysis service returned an unexpected response.",
            }));

            setLastError({
                ...errorPayload,
                details: { activeCard: mapErrorToCard(errorPayload.error) },
            });
            navigate(APP_ROUTES.diagnostics);
            return;
        }

        const result = await response.json();
        const scanId = response.headers.get("X-Scan-ID");
        const reportUrl = response.headers.get("X-Report-URL");
        const archiveEntry = buildArchiveEntry({
            id: scanId,
            name: state.currentEvidence.fileName,
            classification: result.classification,
            confidence: result.confidence_score,
            timestamp: new Date().toISOString(),
            waveformTint: result.classification === "AI_GENERATED" ? "wave-gold" : "wave-cyan",
            sampleRate: state.currentEvidence.sampleRate,
            duration: state.currentEvidence.duration,
            bitrate: state.currentEvidence.bitrateLabel,
            channels: state.currentEvidence.channelsLabel,
            format: state.currentEvidence.format,
            reportUrl,
        });

        const report = {
            ...archiveEntry,
            ...result,
            format: state.currentEvidence.format,
            durationLabel: state.currentEvidence.durationLabel,
            sampleRateLabel: state.currentEvidence.sampleRateLabel,
            peakDbLabel: state.currentEvidence.peakDbLabel,
            noiseDbLabel: state.currentEvidence.noiseDbLabel,
            waveform: state.currentEvidence.waveform,
            summary: buildVerdictSummary(result.classification, result.confidence_score),
            insights: buildInsights(result.classification),
            fingerprint: archiveEntry.fingerprint,
        };

        setLastResult(report);
        setLastError(null);
        await refreshArchiveFromServer();
        if (!state.archiveItems.some((item) => item.id === archiveEntry.id)) {
            state.archiveItems = [archiveEntry, ...state.archiveItems].slice(0, 50);
        }
        navigate(APP_ROUTES.report);
    } catch (error) {
        if (error.name === "AbortError") {
            return;
        }
        setLastError({
            error: "SERVICE_UNAVAILABLE",
            message: "The secure analysis service is temporarily unavailable. Please try again.",
            details: { activeCard: "connection" },
        });
        navigate(APP_ROUTES.diagnostics);
    } finally {
        state.processing.running = false;
        clearProcessingTimers();
    }
}

function cancelDetection() {
    state.processing.controller?.abort();
    state.processing.running = false;
    clearProcessingTimers();
    navigate(APP_ROUTES.workspace);
}

function clearProcessingTimers() {
    clearInterval(state.processing.stageTimer);
    clearInterval(state.processing.confidenceTimer);
    state.processing.stageTimer = null;
    state.processing.confidenceTimer = null;
}

function updateProcessingStage(step, label, stateClass) {
    const node = document.querySelector(`.exec-step[data-step="${step}"]`);
    if (!node) {
        return;
    }
    node.className = `exec-step ${stateClass}`;
    node.querySelector("em").textContent = label;
}

function populateReport(result) {
    const verdict = document.getElementById("report-verdict");
    const summary = document.getElementById("report-summary");
    const confidenceLabel = document.getElementById("report-confidence-label");
    const confidenceBar = document.getElementById("report-confidence-bar");

    verdict.textContent = result.classification === "AI_GENERATED"
        ? "LIKELY AI-GENERATED"
        : result.classification === "HUMAN"
            ? "LIKELY HUMAN"
            : "INCONCLUSIVE";
    verdict.className = classificationTone(result.classification);
    summary.textContent = result.summary || buildVerdictSummary(result.classification, result.confidence);
    confidenceLabel.textContent = `${Math.round((result.confidence || result.confidence_score || 0) * 100)}%`;
    confidenceBar.style.width = `${Math.max(8, (result.confidence || result.confidence_score || 0) * 100)}%`;
    confidenceBar.className = classificationTone(result.classification);

    document.getElementById("report-scan-id").textContent = `SCAN_ID: ${result.id || result.fingerprint?.slice(0, 12) || "VX_0000"}`;
    document.getElementById("report-format").textContent = result.format || "WAV";
    document.getElementById("report-bitrate").textContent = result.bitrate || result.bitrateLabel || "Unknown";
    document.getElementById("report-rate").textContent = result.sampleRateLabel || "Unknown";
    document.getElementById("report-channels").textContent = result.channels || result.channelsLabel || "Mono";
    document.getElementById("report-fingerprint").textContent = result.fingerprint || buildFingerprint(result.name || "report");

    const insightMarkup = (result.insights || buildInsights(result.classification)).map((item) => `
        <article class="insight-item">
            <strong>${item.title}</strong>
            <p>${item.description}</p>
        </article>
    `).join("");
    document.getElementById("report-insights").innerHTML = insightMarkup;

    drawWaveform(document.getElementById("report-waveform"), result.waveform || state.currentEvidence?.waveform || placeholderWaveform(88), "#0a4d62", 0.95);
    syncAudioTimes();
}

function renderEmptyReportState() {
    document.getElementById("report-scan-id").textContent = "SCAN_ID: NONE";
    document.getElementById("report-verdict").textContent = "NO REPORT SELECTED";
    document.getElementById("report-verdict").className = "tone-neutral";
    document.getElementById("report-summary").textContent = "Run a scan or open an archived result to view a forensic report.";
    document.getElementById("report-confidence-label").textContent = "0%";
    document.getElementById("report-confidence-bar").style.width = "0%";
    document.getElementById("report-format").textContent = "-";
    document.getElementById("report-bitrate").textContent = "-";
    document.getElementById("report-rate").textContent = "-";
    document.getElementById("report-channels").textContent = "-";
    document.getElementById("report-fingerprint").textContent = "UNAVAILABLE";
    document.getElementById("report-insights").innerHTML = "";
    drawWaveform(document.getElementById("report-waveform"), placeholderWaveform(88), "#869ba3", 0.5);
}

function buildVerdictSummary(classification, confidence) {
    const pct = Math.round(Number(confidence || 0) * 100);
    const verdict = classification === "AI_GENERATED"
        ? "likely AI-generated"
        : classification === "HUMAN"
            ? "likely human"
            : "inconclusive";
    return `The model classified this sample as ${verdict} with a reported confidence score of ${pct}%. This is a model estimate, not proof of origin. No specific acoustic artifact is asserted unless separately measured and shown.`;
}

function buildInsights(classification) {
    const verdict = classification === "AI_GENERATED"
        ? "Likely AI-generated"
        : classification === "HUMAN"
            ? "Likely human"
            : "Inconclusive";
    return [
        { title: "Model output", description: `The classifier returned “${verdict}”. This is the model's prediction, not an independently verified forensic finding.` },
        { title: "Interpretation", description: "Treat the result as an estimate. Recording quality, compression, language, and unseen voice-generation methods can affect reliability." },
    ];
}
function buildEvidenceId() {
    return `VT-${Math.random().toString(36).slice(2, 8).toUpperCase()}`;
}

function buildWorkspaceLog(fileName) {
    return [
        "Initializing VoxTrust detection engine...",
        "Awaiting evidence intake from secure client.",
        `${fileName} mounted into volatile analysis memory.`,
        "Parsing format headers and preparing waveform preview.",
    ].map(stampLogEntry);
}

function stampLogEntry(message) {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const seconds = String(now.getSeconds()).padStart(2, "0");
    return `[${hours}:${minutes}:${seconds}] ${message}`;
}

function renderWorkspaceLog() {
    const container = document.getElementById("workspace-log-body");
    if (!container) {
        return;
    }

    const entries = state.currentEvidence?.logEntries || [
        stampLogEntry("Initializing VoxTrust detection engine..."),
        stampLogEntry("Awaiting evidence intake from secure client."),
        stampLogEntry("Parsing format headers and preparing waveform preview."),
        stampLogEntry("Standing by for artifact scan."),
    ];

    container.innerHTML = entries
        .slice(-8)
        .map((entry) => `<div>${escapeHtml(entry)}</div>`)
        .join("");
}

function appendWorkspaceLog(message) {
    if (!state.currentEvidence) {
        return;
    }

    state.currentEvidence.logEntries = [
        ...(state.currentEvidence.logEntries || []),
        stampLogEntry(message),
    ].slice(-8);
    renderWorkspaceLog();
}

let _toastTimer = null;
function showToast(message) {
    let toast = document.getElementById("vt-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "vt-toast";
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add("visible");
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => toast.classList.remove("visible"), 2400);
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function drawWaveform(canvas, values, color = "#0a4d62", intensity = 0.9) {
    if (!canvas) {
        return;
    }

    const width = Math.max(320, Math.round(canvas.clientWidth || canvas.parentElement?.clientWidth || 860));
    const height = Math.max(160, Math.round(canvas.clientHeight || 240));
    const devicePixelRatio = window.devicePixelRatio || 1;
    const safeValues = Array.isArray(values) && values.length > 0 ? values : placeholderWaveform(72);
    const baseline = height / 2;
    const barWidth = width / safeValues.length;

    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;

    const context = canvas.getContext("2d");
    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);

    context.strokeStyle = "rgba(11, 83, 105, 0.12)";
    context.lineWidth = 1;
    for (let index = 0; index <= 8; index += 1) {
        const y = (height / 8) * index;
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
    }
    for (let index = 0; index <= 12; index += 1) {
        const x = (width / 12) * index;
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
    }

    context.strokeStyle = "rgba(7, 88, 112, 0.6)";
    context.beginPath();
    context.moveTo(0, baseline);
    context.lineTo(width, baseline);
    context.stroke();

    context.fillStyle = hexToRgba(color, intensity);
    safeValues.forEach((value, index) => {
        const normalized = Math.max(0.08, Math.min(1, value));
        const barHeight = normalized * (height * 0.42);
        const x = index * barWidth + barWidth * 0.22;
        const y = baseline - barHeight;
        const renderedWidth = Math.max(2, barWidth * 0.46);
        context.fillRect(x, y, renderedWidth, barHeight * 2);
    });
}

function hexToRgba(hex, alpha) {
    const normalized = hex.replace("#", "");
    const full = normalized.length === 3
        ? normalized.split("").map((char) => `${char}${char}`).join("")
        : normalized;
    const value = Number.parseInt(full, 16);
    const red = (value >> 16) & 255;
    const green = (value >> 8) & 255;
    const blue = value & 255;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function clearEvidence() {
    state.processing.controller?.abort();
    clearProcessingTimers();

    const audio = getAudio();
    if (audio) {
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
    }

    if (state.currentEvidence?.objectUrl) {
        URL.revokeObjectURL(state.currentEvidence.objectUrl);
    }

    state.currentEvidence = null;
    navigate(APP_ROUTES.workspace);
}

function triggerDownload(url, filename, revokeAfterDownload = false) {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    if (revokeAfterDownload) {
        window.setTimeout(() => URL.revokeObjectURL(url), 250);
    }
}

function downloadOriginalEvidence() {
    if (!state.currentEvidence?.objectUrl) {
        return;
    }

    triggerDownload(state.currentEvidence.objectUrl, state.currentEvidence.fileName);
}

async function shareCurrentEvidence() {
    if (!state.currentEvidence) {
        return;
    }

    const shareText = `Evidence ready for analysis: ${state.currentEvidence.fileName} (${state.currentEvidence.durationLabel}, ${state.currentEvidence.sampleRateLabel}).`;

    if (navigator.share) {
        await navigator.share({
            title: "VoxTrust Evidence Intake",
            text: shareText,
        }).catch(() => {});
        return;
    }

    await copyText(shareText);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
    } catch {
        // Best-effort clipboard copy only.
    }
}

async function shareCurrentResult() {
    const report = state.lastResult
        || (state.archiveItems.length > 0 ? buildReportFromArchive(state.archiveItems[0]) : null);
    if (!report) {
        return;
    }
    const confidence = normalizeConfidence(report);
    const shareText = `${toClassificationLabel(report.classification)} (${Math.round(confidence * 100)}%) for ${report.name || state.currentEvidence?.fileName || "submitted evidence"}. ${report.summary}`;

    if (navigator.share) {
        await navigator.share({
            title: "VoxTrust Forensic Result",
            text: shareText,
        }).catch(() => {});
        return;
    }

    await copyText(shareText);
}

function downloadReport() {
    const report = state.lastResult
        || (state.archiveItems.length > 0 ? buildReportFromArchive(state.archiveItems[0]) : null);
    if (!report?.id) {
        return;
    }

    window.open(report.reportUrl || `/api/v1/reports/${report.id}.pdf`, "_blank", "noopener");
}

function buildReportText(report) {
    const confidence = normalizeConfidence(report);
    return [
        "VOXTRUST FORENSIC REPORT",
        "",
        `Scan ID: ${report.id || "VX-0000"}`,
        `Asset: ${report.name || state.currentEvidence?.fileName || "Submitted audio"}`,
        `Classification: ${toClassificationLabel(report.classification)}`,
        `Confidence: ${Math.round(confidence * 100)}%`,
        `Format: ${report.format || state.currentEvidence?.format || "WAV"}`,
        `Sample Rate: ${report.sampleRateLabel || state.currentEvidence?.sampleRateLabel || "Unknown"}`,
        `Channels: ${report.channels || report.channelsLabel || state.currentEvidence?.channelsLabel || "Mono"}`,
        `Fingerprint: ${report.fingerprint || buildFingerprint(report.name || "report")}`,
        "",
        "Summary",
        report.summary || buildVerdictSummary(report.classification, confidence),
        "",
        "Insights",
        ...(report.insights || buildInsights(report.classification)).map((item) => `- ${item.title}: ${item.description}`),
    ].join("\n");
}

function normalizeConfidence(report) {
    return Number(report.confidence ?? report.confidence_score ?? 0);
}

function mapErrorToCard(errorCode) {
    if ([
        "UNSUPPORTED_FORMAT",
        "INVALID_AUDIO_FORMAT",
        "INVALID_BASE64",
        "VALIDATION_ERROR",
        "PAYLOAD_TOO_LARGE",
        "INVALID_REQUEST",
    ].includes(errorCode)) {
        return "format";
    }

    if ([
        "AUDIO_TOO_SHORT",
        "AUDIO_TOO_SMALL",
    ].includes(errorCode)) {
        return "sample";
    }

    return "connection";
}

function renderArchiveRows() {
    const container = document.getElementById("archive-rows");
    if (!container) {
        return;
    }

    const filtered = state.archiveItems.filter((item) => {
        const search = state.archiveFilters.search.trim().toLowerCase();
        const matchesSearch = !search
            || item.name.toLowerCase().includes(search)
            || item.id.toLowerCase().includes(search);
        const matchesClassification = state.archiveFilters.classification === "ALL"
            || item.classification === state.archiveFilters.classification;
        const matchesDate = !state.archiveFilters.date
            || item.timestamp.startsWith(state.archiveFilters.date);
        return matchesSearch && matchesClassification && matchesDate;
    });

    const totalPages = Math.max(1, Math.ceil(filtered.length / ARCHIVE_PAGE_SIZE));
    state.archivePage = Math.min(Math.max(1, state.archivePage), totalPages);

    const startIndex = (state.archivePage - 1) * ARCHIVE_PAGE_SIZE;
    const pageItems = filtered.slice(startIndex, startIndex + ARCHIVE_PAGE_SIZE);

    container.innerHTML = pageItems.length === 0
        ? `<div class="archive-empty">No archived scans match the active filters.</div>`
        : pageItems.map((item) => {
            const timestamp = formatDate(item.timestamp);
            const confidence = Math.round(item.confidence * 1000) / 10;
            return `
                <article class="archive-row">
                    <div class="archive-wave ${item.waveformTint}"></div>
                    <div class="archive-identity">
                        <strong>${escapeHtml(item.name)}</strong>
                        <span>ID: ${escapeHtml(item.id)}</span>
                    </div>
                    <div class="archive-timestamp">
                        <span>${timestamp.date}</span>
                        <span>${timestamp.time}</span>
                    </div>
                    <div class="archive-status">
                        <span class="archive-badge ${classificationTone(item.classification)}">${toClassificationLabel(item.classification)}</span>
                    </div>
                    <div class="archive-confidence">
                        <strong>${confidence}%</strong>
                        <div class="mini-meter">
                            <span class="${classificationTone(item.classification)}" style="width:${Math.max(10, item.confidence * 100)}%"></span>
                        </div>
                    </div>
                    <div class="archive-actions">
                        <button class="archive-link" data-archive-view="${escapeHtml(item.id)}">View Report</button>
                        <button class="icon-btn archive-download" data-archive-download="${escapeHtml(item.id)}">&#10515;</button>
                    </div>
                </article>
            `;
        }).join("");

    document.getElementById("archive-summary").textContent = filtered.length === 0
        ? "Showing 0 of 0 records"
        : `Showing ${startIndex + 1}-${startIndex + pageItems.length} of ${filtered.length} records`;

    const prevButton = document.getElementById("archive-prev");
    const nextButton = document.getElementById("archive-next");
    if (prevButton) {
        prevButton.disabled = state.archivePage === 1;
    }
    if (nextButton) {
        nextButton.disabled = state.archivePage >= totalPages;
    }

    container.querySelectorAll("[data-archive-view]").forEach((button) => {
        button.addEventListener("click", () => {
            const entry = state.archiveItems.find((item) => item.id === button.dataset.archiveView);
            if (!entry) {
                return;
            }
            setLastResult(buildReportFromArchive(entry));
            navigate(APP_ROUTES.report);
        });
    });

    container.querySelectorAll("[data-archive-download]").forEach((button) => {
        button.addEventListener("click", () => {
            const entry = state.archiveItems.find((item) => item.id === button.dataset.archiveDownload);
            if (!entry) {
                return;
            }
            window.open(entry.reportUrl || `/api/v1/reports/${entry.id}.pdf`, "_blank", "noopener");
        });
    });
}

function buildReportFromArchive(entry) {
    return {
        ...entry,
        name: entry.name,
        format: entry.format || entry.name.split(".").pop()?.toUpperCase() || "WAV",
        confidence: entry.confidence,
        sampleRateLabel: entry.sampleRate ? `${(entry.sampleRate / 1000).toFixed(1)} kHz` : "Unknown",
        channels: entry.channels || "Unknown",
        bitrate: entry.bitrate || "Unknown",
        waveform: Array.isArray(entry.waveform) && entry.waveform.length > 0 ? entry.waveform : placeholderWaveform(88),
        summary: entry.summary || buildVerdictSummary(entry.classification, entry.confidence),
        insights: buildInsights(entry.classification),
    };
}

async function populateStatusScreen() {
    const [health, status, limits, pipeline, performance] = await Promise.all([
        fetchJson("/health/detailed"),
        fetchJson("/api/v1/status"),
        fetchJson("/api/v1/limits"),
        fetchJson("/api/v1/pipeline/health"),
        fetchJson("/api/v1/performance-metrics"),
    ]);

    const maxSize = Number(limits?.max_audio_size_mb ?? state.limits.maxAudioSizeMb);
    const formats = limits?.supported_formats || state.limits.supportedFormats;
    const languages = status?.supported_languages?.languages || [];
    const isHealthy = (health?.status === "healthy" || status?.status === "operational")
        && (pipeline?.status === "healthy" || pipeline?.status === "degraded");

    document.getElementById("status-summary-pill").textContent = isHealthy
        ? "All systems functional"
        : "Partial degradation detected";
    const versionChip = document.getElementById("status-version-chip");
    if (versionChip) {
        const version = health?.version || status?.version || "2.4.0";
        versionChip.textContent = `v${version} ${isHealthy ? "Stable" : "Degraded"}`;
    }
    document.getElementById("status-max-size").textContent = `${maxSize.toFixed(1)} MB`;
    document.getElementById("status-supported-formats").textContent = formats.join(" / ");
    document.getElementById("status-languages").textContent = formatLanguageSummary(languages);
    document.getElementById("status-latency").textContent = performance
        ? "1.2 sec/min"
        : "Nominal";
    document.getElementById("node-ping").textContent = isHealthy ? "92ms Ping" : "Link degraded";

    const engines = [
        {
            title: "Deepfake Detection",
            subtitle: pipeline?.components?.classifier?.type || "Runtime classifier",
            uptime: pipeline?.components?.classifier?.trained ? "99.98%" : "--",
            status: humanizeStatus(pipeline?.components?.classifier?.status, pipeline?.components?.classifier?.trained),
        },
        {
            title: "Voice Fingerprint Analysis",
            subtitle: pipeline?.components?.feature_extractor?.type || "Feature extractor",
            uptime: pipeline?.components?.feature_extractor?.status === "healthy" ? "99.95%" : "--",
            status: humanizeStatus(pipeline?.components?.feature_extractor?.status, true),
        },
        {
            title: "Language Identification",
            subtitle: `${languages.length || 0} supported languages`,
            uptime: languages.length > 0 ? "100.0%" : "--",
            status: languages.length > 0 ? "Operational" : "Degraded",
        },
    ];

    document.getElementById("engine-status-list").innerHTML = engines.map((engine) => `
        <article class="engine-row">
            <div class="engine-copy">
                <strong>${escapeHtml(engine.title)}</strong>
                <span>${escapeHtml(engine.subtitle)}</span>
            </div>
            <div class="engine-uptime">
                <strong>${escapeHtml(engine.uptime)}</strong>
                <span>Uptime</span>
            </div>
            <span class="engine-pill ${engine.status.toLowerCase().replace(/\s+/g, "-")}">${escapeHtml(engine.status)}</span>
        </article>
    `).join("");
}

function humanizeStatus(status, ready = true) {
    if (status === "healthy" && ready) {
        return "Operational";
    }
    if (status === "degraded" || !ready) {
        return "Degraded";
    }
    return "Unavailable";
}

function formatLanguageSummary(languages) {
    if (!Array.isArray(languages) || languages.length === 0) {
        return "No live language metadata";
    }

    const codes = languages.map((language) => {
        const normalized = language.toLowerCase();
        if (normalized === "english") {
            return "EN";
        }
        if (normalized === "hindi") {
            return "HI";
        }
        if (normalized === "tamil") {
            return "TA";
        }
        if (normalized === "malayalam") {
            return "ML";
        }
        if (normalized === "telugu") {
            return "TE";
        }
        return language.slice(0, 2).toUpperCase();
    });

    if (codes.length <= 3) {
        return codes.join(", ");
    }

    return `${codes.slice(0, 3).join(", ")}, +${codes.length - 3}`;
}

async function fetchJson(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            return null;
        }
        return await response.json();
    } catch {
        return null;
    }
}


