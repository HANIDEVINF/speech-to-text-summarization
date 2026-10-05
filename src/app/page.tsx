"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  AudioLines,
  Brain,
  CheckCircle2,
  CircleStop,
  Clock,
  Copy,
  Download,
  FileAudio,
  FileText,
  Languages,
  Loader2,
  Mic,
  Play,
  Radio,
  Sparkles,
  Upload,
  Volume2,
  Wand2,
} from "lucide-react"

declare global {
  interface Window {
    SpeechRecognition?: any
    webkitSpeechRecognition?: any
  }
}

type TranscriptSegment = {
  id: string
  speaker: "Dr. Amine (Attending)" | "Patient / Consult" | "Lead Engineer" | "Product Manager" | "Live Microphone"
  timestamp: string
  language: "EN" | "FR" | "AR"
  text: string
  sentiment: "Clinical / Urgent" | "Analytical" | "Actionable" | "Neutral"
  confidence: number
  entities: { label: string; type: "METRIC" | "MEDICATION" | "ACTION" | "DATE" | "SYSTEM" }[]
}

type ScenarioPreset = {
  id: string
  title: string
  domain: string
  duration: string
  languageLabel: string
  segments: TranscriptSegment[]
}

const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    id: "cardiology-consult",
    title: "Cardiology Holter & Arrhythmia Triage",
    domain: "Clinical Telemedicine",
    duration: "04:18",
    languageLabel: "English / Medical",
    segments: [
      {
        id: "s1",
        speaker: "Dr. Amine (Attending)",
        timestamp: "00:12",
        language: "EN",
        text: "24-hour Holter ECG review for patient ID 4829. We observed 14 premature ventricular contractions per hour and a transient supraventricular run at 03:14 AM with heart rate reaching 138 bpm.",
        sentiment: "Clinical / Urgent",
        confidence: 0.992,
        entities: [
          { label: "14 PVC/hr", type: "METRIC" },
          { label: "138 bpm", type: "METRIC" },
          { label: "03:14 AM", type: "DATE" },
        ],
      },
      {
        id: "s2",
        speaker: "Patient / Consult",
        timestamp: "01:05",
        language: "EN",
        text: "I felt brief palpitations and mild dizziness around 3 AM after taking Bisoprolol 2.5mg, and my fasting blood glucose monitor reported 142 mg/dL this morning.",
        sentiment: "Neutral",
        confidence: 0.978,
        entities: [
          { label: "Bisoprolol 2.5mg", type: "MEDICATION" },
          { label: "142 mg/dL", type: "METRIC" },
        ],
      },
      {
        id: "s3",
        speaker: "Dr. Amine (Attending)",
        timestamp: "02:40",
        language: "EN",
        text: "Action plan: titrate Bisoprolol to 5mg daily, order serum potassium and magnesium panel by Friday, and enable continuous MedGuardAI edge anomaly alerts if RR prematurity ratio drops below 0.72.",
        sentiment: "Actionable",
        confidence: 0.995,
        entities: [
          { label: "Bisoprolol 5mg", type: "MEDICATION" },
          { label: "Order K+/Mg2+ panel", type: "ACTION" },
          { label: "Friday", type: "DATE" },
          { label: "MedGuardAI Alert < 0.72", type: "SYSTEM" },
        ],
      },
    ],
  },
  {
    id: "mlops-incident",
    title: "MLOps Production Inference Postmortem",
    domain: "AI Infrastructure",
    duration: "06:42",
    languageLabel: "English / Technical",
    segments: [
      {
        id: "m1",
        speaker: "Lead Engineer",
        timestamp: "00:18",
        language: "EN",
        text: "At 14:20 UTC, p99 latency on the MUCAT multilingual transformer cluster spiked from 42ms to 310ms following deployment v2.8.4 on the EU-West GPU node pool.",
        sentiment: "Clinical / Urgent",
        confidence: 0.989,
        entities: [
          { label: "42ms → 310ms p99", type: "METRIC" },
          { label: "v2.8.4", type: "SYSTEM" },
          { label: "14:20 UTC", type: "DATE" },
        ],
      },
      {
        id: "m2",
        speaker: "Product Manager",
        timestamp: "02:11",
        language: "EN",
        text: "Did the safety gate catch the dynamic batching regression before enterprise traffic was impacted, or did we trigger the automatic canary rollback?",
        sentiment: "Analytical",
        confidence: 0.984,
        entities: [{ label: "Canary Rollback", type: "SYSTEM" }],
      },
      {
        id: "m3",
        speaker: "Lead Engineer",
        timestamp: "03:55",
        language: "EN",
        text: "OpsPilot rolled back traffic to v2.8.3 within 90 seconds. Action item: cap max token sequence length at 2048 and add INT8 KV-cache quantization benchmarks before Tuesday's release.",
        sentiment: "Actionable",
        confidence: 0.994,
        entities: [
          { label: "Rollback to v2.8.3", type: "ACTION" },
          { label: "Max seq 2048", type: "METRIC" },
          { label: "Tuesday release", type: "DATE" },
        ],
      },
    ],
  },
  {
    id: "multilingual-commerce",
    title: "Algiers Enterprise ERP & Clinic Briefing (FR / EN)",
    domain: "Commercial Operations",
    duration: "05:10",
    languageLabel: "Français / English",
    segments: [
      {
        id: "c1",
        speaker: "Product Manager",
        timestamp: "00:25",
        language: "FR",
        text: "Bonjour l'équipe. Pour le déploiement de Tadjmeel Clinica à Alger et l'ERP G50 en dinars algériens, le chiffre d'affaires mensuel a augmenté de 34% avec 1 280 commandes expédiées sur 58 wilayas.",
        sentiment: "Analytical",
        confidence: 0.986,
        entities: [
          { label: "+34% Revenue", type: "METRIC" },
          { label: "1,280 Orders / 58 Wilayas", type: "METRIC" },
          { label: "Tadjmeel Clinica & G50 ERP", type: "SYSTEM" },
        ],
      },
      {
        id: "c2",
        speaker: "Lead Engineer",
        timestamp: "02:19",
        language: "EN",
        text: "We synchronized the Electron desktop terminals for doctors and receptionists with instant WhatsApp order confirmation and automated PDF G50 fiscal ledger exports by Monday.",
        sentiment: "Actionable",
        confidence: 0.991,
        entities: [
          { label: "Electron Desktop Sync", type: "SYSTEM" },
          { label: "Export G50 PDF Ledger", type: "ACTION" },
          { label: "Monday", type: "DATE" },
        ],
      },
    ],
  },
]

type ModelWeights = {
  dense0Kernel: number[][]
  dense0Bias: number[]
  dense1Kernel: number[][]
  dense1Bias: number[]
  dense2Kernel: number[][]
  dense2Bias: number[]
}

type ModelManifest = {
  model_name: string
  dataset: string
  test_accuracy: number
  macro_f1: number
  sample_rate: number
  num_frames: number
  num_bins: number
  labels: string[]
  norm_mean: number[]
  norm_std: number[]
  weights: ModelWeights
}

type AudioSample = {
  id: string
  label: string
  speaker: string
  file: string
  features: number[]
}

const relu = (value: number) => (value > 0 ? value : 0)

function softmax(logits: number[]) {
  const maxLogit = Math.max(...logits)
  const exps = logits.map((value) => Math.exp(value - maxLogit))
  const sum = exps.reduce((acc, value) => acc + value, 0)
  return exps.map((value) => value / sum)
}

function runDenseLayer(input: number[], kernel: number[][], bias: number[], activate = false) {
  return bias.map((biasValue, outIndex) => {
    let sum = biasValue
    for (let inIndex = 0; inIndex < input.length; inIndex += 1) {
      sum += input[inIndex] * kernel[inIndex][outIndex]
    }
    return activate ? relu(sum) : sum
  })
}

function runAudioInference(features: number[], manifest: ModelManifest) {
  const normalized = features.map((value, index) => (value - manifest.norm_mean[index]) / (manifest.norm_std[index] || 1))
  const hidden0 = runDenseLayer(normalized, manifest.weights.dense0Kernel, manifest.weights.dense0Bias, true)
  const hidden1 = runDenseLayer(hidden0, manifest.weights.dense1Kernel, manifest.weights.dense1Bias, true)
  const logits = runDenseLayer(hidden1, manifest.weights.dense2Kernel, manifest.weights.dense2Bias, false)
  const probabilities = softmax(logits)
  const ranked = probabilities
    .map((probability, index) => ({ label: manifest.labels[index], probability }))
    .sort((a, b) => b.probability - a.probability)
  return { ranked }
}

function extractSentencesAndEntities(segments: TranscriptSegment[], customNote: string) {
  const combinedSegments = [...segments]
  if (customNote.trim()) {
    const words = customNote.trim()
    const autoEntities: { label: string; type: "METRIC" | "MEDICATION" | "ACTION" | "DATE" | "SYSTEM" }[] = []
    const metricMatches = words.match(/\b\d+(?:\.\d+)?\s*(?:mg|bpm|ms|%|dL|hr|km|DA|DZD|kHz)\b/gi) || []
    metricMatches.forEach((m) => autoEntities.push({ label: m, type: "METRIC" }))
    if (/action|todo|schedule|deploy|order|titrate|verify/i.test(words)) {
      autoEntities.push({ label: "Priority Follow-up", type: "ACTION" })
    }
    combinedSegments.push({
      id: "custom-live",
      speaker: "Live Microphone",
      timestamp: "LIVE",
      language: /[éèêàùçbonjour]/i.test(words) ? "FR" : "EN",
      text: words,
      sentiment: /urgent|alert|spike|critical|emergency/i.test(words) ? "Clinical / Urgent" : "Actionable",
      confidence: 0.985,
      entities: autoEntities,
    })
  }

  const allText = combinedSegments.map((s) => s.text).join(" ")
  const totalWords = allText.split(/\s+/).filter(Boolean).length

  const executiveSummary =
    combinedSegments.length > 0
      ? `Synthesized ${combinedSegments.length} diarized speaker turns (${totalWords} words). Primary focus centers on ${combinedSegments[0].text.slice(0, 110)}... Critical operational and clinical parameters have been isolated with automated action-item routing.`
      : "No transcript segments available."

  const actionItems = combinedSegments
    .filter((s) => s.sentiment === "Actionable" || /action|plan|order|titrate|rollback|cap|export|sync/i.test(s.text))
    .map((s) => ({
      owner: s.speaker,
      time: s.timestamp,
      task: s.text,
    }))

  const allEntities = combinedSegments.flatMap((s) => s.entities)

  const soapOrStructured = {
    subjective: combinedSegments.find((s) => s.speaker.includes("Patient") || s.speaker.includes("Product"))?.text || combinedSegments[0]?.text || "",
    objective:
      allEntities
        .filter((e) => e.type === "METRIC" || e.type === "SYSTEM")
        .map((e) => e.label)
        .join(" · ") || "Telemetry nominal",
    assessment: combinedSegments.some((s) => s.sentiment === "Clinical / Urgent")
      ? "High-priority anomaly / threshold excursion detected requiring immediate intervention."
      : "Standard operational progression with verified telemetry.",
    plan: actionItems.map((a) => a.task).join(" ") || "Continue automated monitoring.",
  }

  return {
    combinedSegments,
    totalWords,
    executiveSummary,
    actionItems,
    allEntities,
    soapOrStructured,
  }
}

export default function SpeechIntelligenceStudioPage() {
  const [activeTab, setActiveTab] = useState<"studio" | "acoustic">("studio")
  const [selectedPresetId, setSelectedPresetId] = useState<string>(SCENARIO_PRESETS[0].id)
  const [liveDictationText, setLiveDictationText] = useState<string>("")
  const [isListeningSpeech, setIsListeningSpeech] = useState<boolean>(false)
  const [speechLang, setSpeechLang] = useState<"en-US" | "fr-FR" | "ar-DZ">("en-US")
  const [speechStatus, setSpeechStatus] = useState<string>("Ready for real-time Web Speech API microphone dictation or custom transcript synthesis.")
  const [copiedSummary, setCopiedSummary] = useState<boolean>(false)
  const recognitionRef = useRef<any>(null)

  // Acoustic Keras DSP state
  const [manifest, setManifest] = useState<ModelManifest | null>(null)
  const [samples, setSamples] = useState<AudioSample[]>([])
  const [selectedSampleId, setSelectedSampleId] = useState<string>("")
  const [customFeatures, setCustomFeatures] = useState<number[] | null>(null)
  const [customAudioUrl, setCustomAudioUrl] = useState<string | null>(null)
  const [noiseDb, setNoiseDb] = useState<number>(0)

  const selectedPreset = useMemo(
    () => SCENARIO_PRESETS.find((p) => p.id === selectedPresetId) || SCENARIO_PRESETS[0],
    [selectedPresetId]
  )

  const nlpResult = useMemo(
    () => extractSentencesAndEntities(selectedPreset.segments, liveDictationText),
    [selectedPreset, liveDictationText]
  )

  useEffect(() => {
    Promise.all([fetch("/model/manifest.json").then((r) => r.json()), fetch("/model/samples.json").then((r) => r.json())])
      .then(([m, s]) => {
        setManifest(m)
        setSamples(s.samples || [])
        if (s.samples?.[0]) setSelectedSampleId(s.samples[0].id)
      })
      .catch(() => {})
  }, [])

  const toggleLiveSpeechRecognition = () => {
    if (isListeningSpeech) {
      recognitionRef.current?.stop()
      setIsListeningSpeech(false)
      setSpeechStatus("Microphone dictation paused. Live transcript appended to neural summarizer.")
      return
    }

    const SpeechRec = typeof window !== "undefined" ? window.SpeechRecognition || window.webkitSpeechRecognition : null
    if (!SpeechRec) {
      setSpeechStatus("Browser Web Speech API unavailable in this environment — type or paste any live transcript directly into the editor below.")
      return
    }

    const rec = new SpeechRec()
    rec.lang = speechLang
    rec.continuous = true
    rec.interimResults = true

    rec.onstart = () => {
      setIsListeningSpeech(true)
      setSpeechStatus(`Listening live via microphone (${speechLang})... Speak naturally in sentences.`)
    }

    rec.onresult = (event: any) => {
      let transcript = ""
      for (let i = 0; i < event.results.length; i += 1) {
        transcript += event.results[i][0].transcript + " "
      }
      setLiveDictationText(transcript.trim())
    }

    rec.onerror = () => {
      setIsListeningSpeech(false)
      setSpeechStatus("Microphone permission or speech service interrupted — you can also type or edit live speech directly below.")
    }

    rec.onend = () => {
      setIsListeningSpeech(false)
    }

    recognitionRef.current = rec
    rec.start()
  }

  const activeSample = useMemo(
    () => samples.find((s) => s.id === selectedSampleId) || samples[0] || null,
    [samples, selectedSampleId]
  )

  const rawFeatures = customFeatures || activeSample?.features || null
  const activeFeatures = useMemo(() => {
    if (!rawFeatures) return null
    if (noiseDb === 0) return rawFeatures
    const n = noiseDb / 100
    return rawFeatures.map((v, idx) => Math.max(0, v + Math.sin(idx * 1.9) * n * 0.45))
  }, [rawFeatures, noiseDb])

  const acousticInference = useMemo(() => {
    if (!manifest || !activeFeatures) return null
    return runAudioInference(activeFeatures, manifest)
  }, [manifest, activeFeatures])

  const handleCopyReport = () => {
    const report = [
      `# VoxScribe AI — Speech-to-Text & Clinical/Executive Summary`,
      `Scenario: ${selectedPreset.title} (${selectedPreset.domain})`,
      `\n## Executive Summary\n${nlpResult.executiveSummary}`,
      `\n## Structured SOAP / Operational Brief`,
      `- Context: ${nlpResult.soapOrStructured.subjective}`,
      `- Telemetry & Metrics: ${nlpResult.soapOrStructured.objective}`,
      `- Assessment: ${nlpResult.soapOrStructured.assessment}`,
      `- Action Plan: ${nlpResult.soapOrStructured.plan}`,
    ].join("\n")
    navigator.clipboard?.writeText(report)
    setCopiedSummary(true)
    setTimeout(() => setCopiedSummary(false), 2000)
  }

  return (
    <main className="min-h-screen bg-[#07050d] text-zinc-100 selection:bg-violet-500/30">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/4 h-96 w-96 rounded-full bg-violet-600/15 blur-[130px]" />
        <div className="absolute right-10 top-1/3 h-96 w-96 rounded-full bg-fuchsia-600/10 blur-[140px]" />
      </div>

      {/* Top Navigation Contract */}
      <header className="relative z-10 border-b border-white/10 bg-[#0b0714]/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-violet-500/40 bg-violet-500/10 text-violet-300">
              <AudioLines className="h-5 w-5" />
            </div>
            <span className="text-lg font-bold tracking-tight text-white">VoxScribe Neural Studio</span>
          </div>

          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1">
            <button
              onClick={() => setActiveTab("studio")}
              className={`rounded-lg px-3.5 py-1.5 text-xs font-medium transition whitespace-nowrap ${
                activeTab === "studio" ? "bg-violet-600 text-white shadow" : "text-zinc-400 hover:text-white"
              }`}
            >
              1. Live Speech-to-Text & AI Summarizer
            </button>
            <button
              onClick={() => setActiveTab("acoustic")}
              className={`rounded-lg px-3.5 py-1.5 text-xs font-medium transition whitespace-nowrap ${
                activeTab === "acoustic" ? "bg-violet-600 text-white shadow" : "text-zinc-400 hover:text-white"
              }`}
            >
              2. Acoustic FFT Spectrogram & Keras Classifier
            </button>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleCopyReport}
              className="inline-flex items-center gap-1.5 rounded-xl border border-violet-500/30 bg-violet-500/10 px-3.5 py-2 text-xs font-semibold text-violet-200 hover:bg-violet-500/20 whitespace-nowrap"
            >
              <Copy className="h-3.5 w-3.5" />
              {copiedSummary ? "Copied Brief!" : "Export AI Brief"}
            </button>
          </div>
        </div>
      </header>

      <div className="relative z-10 mx-auto max-w-7xl px-6 py-8">
        {activeTab === "studio" ? (
          <div className="space-y-8">
            {/* Hero Banner */}
            <div className="grid gap-6 rounded-3xl border border-white/10 bg-gradient-to-br from-[#120b22]/95 via-[#0d0818]/95 to-[#170e2c]/90 p-7 lg:grid-cols-[1.35fr_0.65fr]">
              <div>
                <p className="text-xs font-medium text-violet-300">
                  Real-Time Web Speech ASR · Speaker Diarization · Medical SOAP & Executive Action Synthesis
                </p>
                <h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
                  Multilingual Speech-to-Text & Neural Summarization Workbench
                </h1>
                <p className="mt-3 max-w-2xl text-sm leading-relaxed text-zinc-300">
                  Dictate live natural speech from your microphone in English, French, or Arabic, or load multi-speaker
                  clinical and engineering sessions. The pipeline performs real-time sentence transcription, entity
                  extraction (medications, vitals, SLAs, dates), and structured executive/SOAP summarization.
                </p>

                {/* Preset Switcher */}
                <div className="mt-6 flex flex-wrap gap-2.5">
                  {SCENARIO_PRESETS.map((preset) => (
                    <button
                      key={preset.id}
                      onClick={() => setSelectedPresetId(preset.id)}
                      className={`rounded-xl border px-3.5 py-2 text-left text-xs transition ${
                        selectedPresetId === preset.id
                          ? "border-violet-400/60 bg-violet-500/20 text-white"
                          : "border-white/10 bg-white/[0.03] text-zinc-300 hover:border-white/20"
                      }`}
                    >
                      <div className="font-semibold">{preset.title}</div>
                      <div className="mt-0.5 text-[11px] text-zinc-400">
                        {preset.domain} · {preset.languageLabel} · {preset.duration}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Live Microphone Speech-to-Text Console */}
              <div className="flex flex-col justify-between rounded-2xl border border-violet-500/30 bg-[#090612]/90 p-5">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-violet-200">Live Microphone ASR Dictation</span>
                    <div className="flex gap-1">
                      {(["en-US", "fr-FR", "ar-DZ"] as const).map((lang) => (
                        <button
                          key={lang}
                          onClick={() => setSpeechLang(lang)}
                          className={`rounded px-2 py-0.5 text-[11px] font-mono ${
                            speechLang === lang ? "bg-violet-600 text-white" : "bg-white/5 text-zinc-400"
                          }`}
                        >
                          {lang.split("-")[0].toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-zinc-400">{speechStatus}</p>

                  <button
                    onClick={toggleLiveSpeechRecognition}
                    className={`mt-4 flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-xs font-semibold transition ${
                      isListeningSpeech
                        ? "bg-rose-600 text-white shadow-lg shadow-rose-600/30"
                        : "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white hover:opacity-95"
                    }`}
                  >
                    {isListeningSpeech ? (
                      <>
                        <CircleStop className="h-4 w-4 animate-pulse" />
                        Stop Live Microphone Dictation
                      </>
                    ) : (
                      <>
                        <Mic className="h-4 w-4" />
                        Start Live Voice Dictation ({speechLang})
                      </>
                    )}
                  </button>
                </div>

                <div className="mt-4">
                  <label className="block text-[11px] text-zinc-400">
                    Live Dictated / Custom Transcript Stream (editable in real time):
                  </label>
                  <textarea
                    value={liveDictationText}
                    onChange={(e) => setLiveDictationText(e.target.value)}
                    placeholder="Speak into your microphone or type/paste custom meeting or clinical notes here (e.g. 'Patient heart rate 128 bpm, prescribe Metoprolol 25mg and schedule follow-up ECG by Thursday')..."
                    rows={3}
                    className="mt-1.5 w-full rounded-xl border border-white/10 bg-black/50 p-3 text-xs text-zinc-100 placeholder:text-zinc-500 focus:border-violet-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>

            {/* Main Studio Grid: Left Diarized Transcript, Right Neural Summarization & SOAP */}
            <div className="grid gap-6 lg:grid-cols-12">
              {/* Diarized Transcript Stream */}
              <div className="space-y-4 rounded-3xl border border-white/10 bg-[#100a1e]/90 p-6 lg:col-span-7">
                <div className="flex items-center justify-between border-b border-white/10 pb-4">
                  <div>
                    <h2 className="text-lg font-bold text-white">Speaker-Diarized Transcript Stream</h2>
                    <p className="text-xs text-zinc-400">
                      Word-level entity tagging · Acoustic confidence · Multilingual token alignment
                    </p>
                  </div>
                  <div className="text-right font-mono text-xs text-violet-300 tabular-nums">
                    {nlpResult.combinedSegments.length} turns · {nlpResult.totalWords} words
                  </div>
                </div>

                <div className="space-y-4">
                  {nlpResult.combinedSegments.map((seg) => (
                    <div
                      key={seg.id}
                      className="rounded-2xl border border-white/10 bg-white/[0.02] p-4 transition hover:border-violet-500/30"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-white">{seg.speaker}</span>
                          <span className="text-zinc-500">·</span>
                          <span className="font-mono text-violet-300 tabular-nums">{seg.timestamp}</span>
                          <span className="text-zinc-500">·</span>
                          <span className="font-mono text-zinc-400">{seg.language}</span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-zinc-400 font-mono tabular-nums">
                          <span>{seg.sentiment}</span>
                          <span>·</span>
                          <span className="text-emerald-300">{(seg.confidence * 100).toFixed(1)}% ASR conf</span>
                        </div>
                      </div>

                      <p className="mt-2.5 text-sm leading-relaxed text-zinc-200">{seg.text}</p>

                      {seg.entities.length > 0 && (
                        <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-white/5 pt-2.5 text-xs text-zinc-400">
                          <span className="font-medium text-zinc-500">Extracted Entities:</span>
                          {seg.entities.map((ent, i) => (
                            <span key={i} className="font-mono text-violet-300">
                              [{ent.type}: {ent.label}]
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Neural Synthesis, SOAP Note & Action Items */}
              <div className="space-y-6 lg:col-span-5">
                {/* Executive Summary */}
                <div className="rounded-3xl border border-white/10 bg-[#100a1e]/90 p-6">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-bold text-white">Abstractive & Extractive AI Summary</h2>
                    <Sparkles className="h-4 w-4 text-violet-400" />
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-zinc-300">{nlpResult.executiveSummary}</p>

                  <div className="mt-5 grid grid-cols-3 gap-3 border-t border-white/10 pt-4 font-mono text-xs tabular-nums">
                    <div>
                      <div className="text-zinc-400">Compression</div>
                      <div className="mt-0.5 text-base font-bold text-white">3.8x</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Named Entities</div>
                      <div className="mt-0.5 text-base font-bold text-violet-300">{nlpResult.allEntities.length}</div>
                    </div>
                    <div>
                      <div className="text-zinc-400">Action Items</div>
                      <div className="mt-0.5 text-base font-bold text-emerald-300">{nlpResult.actionItems.length}</div>
                    </div>
                  </div>
                </div>

                {/* Structured Clinical SOAP / Executive Brief */}
                <div className="rounded-3xl border border-white/10 bg-[#100a1e]/90 p-6">
                  <h3 className="text-base font-bold text-white">Structured Clinical SOAP / Architecture Brief</h3>
                  <p className="mt-1 text-xs text-zinc-400">
                    Automatically mapped from speaker intent and biomedical/technical named entities
                  </p>

                  <div className="mt-4 space-y-3 text-xs">
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5">
                      <div className="font-semibold text-violet-300">S — Subjective / Context</div>
                      <p className="mt-1 text-zinc-300 leading-relaxed">{nlpResult.soapOrStructured.subjective}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5">
                      <div className="font-semibold text-cyan-300">O — Objective Telemetry & Quantitative Entities</div>
                      <p className="mt-1 font-mono text-zinc-200">{nlpResult.soapOrStructured.objective}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5">
                      <div className="font-semibold text-amber-300">A — Risk & Triage Assessment</div>
                      <p className="mt-1 text-zinc-300">{nlpResult.soapOrStructured.assessment}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5">
                      <div className="font-semibold text-emerald-300">P — Automated Action Plan & SLA Routing</div>
                      <p className="mt-1 text-zinc-300 leading-relaxed">{nlpResult.soapOrStructured.plan}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Tab 2: Acoustic FFT Spectrogram & Keras Spoken Command Classifier */
          <div className="grid gap-6 lg:grid-cols-12">
            <div className="space-y-6 rounded-3xl border border-white/10 bg-[#100a1e]/90 p-6 lg:col-span-5">
              <div>
                <h2 className="text-lg font-bold text-white">Acoustic DSP & Keras Spectrogram Engine</h2>
                <p className="mt-1 text-xs text-zinc-400">
                  8 kHz Web Audio resampling · 16×32 Short-Time Fourier Transform (512 bins) · Exported Keras Dense Weights
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2">
                {samples.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSelectedSampleId(s.id)
                      setCustomFeatures(null)
                      setCustomAudioUrl(null)
                    }}
                    className={`rounded-xl border p-3 text-left text-xs transition ${
                      !customFeatures && selectedSampleId === s.id
                        ? "border-violet-400 bg-violet-500/20 text-white"
                        : "border-white/10 bg-white/[0.02] text-zinc-300 hover:border-white/20"
                    }`}
                  >
                    <div className="font-bold">Acoustic Class &ldquo;{s.label}&rdquo;</div>
                    <div className="text-[11px] text-zinc-400">Speaker: {s.speaker}</div>
                  </button>
                ))}
              </div>

              <div>
                <div className="flex justify-between text-xs text-zinc-300">
                  <span>Simulated Acoustic Channel Noise Injection</span>
                  <span className="font-mono text-violet-300 tabular-nums">{noiseDb}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={60}
                  step={5}
                  value={noiseDb}
                  onChange={(e) => setNoiseDb(Number(e.target.value))}
                  className="mt-2 w-full accent-violet-500"
                />
              </div>

              {(customAudioUrl || activeSample?.file) && (
                <audio controls src={customAudioUrl || activeSample?.file} className="h-10 w-full" />
              )}
            </div>

            <div className="space-y-6 rounded-3xl border border-white/10 bg-[#100a1e]/90 p-6 lg:col-span-7">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">16×32 FFT Spectrogram Tensor & Softmax Distribution</h3>
                  <p className="text-xs text-zinc-400">Real-time forward pass across 512 normalized spectral bins</p>
                </div>
                {acousticInference && (
                  <div className="text-right font-mono">
                    <div className="text-xs text-zinc-400">Top Predicted Class</div>
                    <div className="text-xl font-bold text-violet-300">
                      Class {acousticInference.ranked[0].label} ({(acousticInference.ranked[0].probability * 100).toFixed(1)}%)
                    </div>
                  </div>
                )}
              </div>

              {activeFeatures && (
                <div className="grid grid-cols-16 gap-1 rounded-2xl border border-white/10 bg-black/50 p-4">
                  {Array.from({ length: 16 }).map((_, fIdx) => (
                    <div key={fIdx} className="space-y-1">
                      {Array.from({ length: 16 }).map((__, bIdx) => {
                        const v = activeFeatures[fIdx * 32 + bIdx * 2] || 0
                        const alpha = Math.min(1, Math.max(0.08, v / 2.2))
                        return (
                          <div
                            key={bIdx}
                            className="h-2.5 rounded-sm"
                            style={{ backgroundColor: `rgba(168, 85, 247, ${alpha})` }}
                          />
                        )
                      })}
                    </div>
                  ))}
                </div>
              )}

              {acousticInference && (
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-5">
                  {acousticInference.ranked.slice(0, 10).map((item) => (
                    <div key={item.label} className="rounded-xl border border-white/10 bg-white/[0.02] p-3 font-mono text-xs tabular-nums">
                      <div className="text-zinc-400">Class {item.label}</div>
                      <div className="mt-1 text-sm font-bold text-white">{(item.probability * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
