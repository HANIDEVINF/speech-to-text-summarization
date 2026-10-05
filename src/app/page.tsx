"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  AudioLines,
  BarChart3,
  Brain,
  CircleStop,
  FileAudio,
  Loader2,
  Mic,
  Play,
  Sparkles,
  Upload,
  Volume2,
  Waves,
} from "lucide-react"

type AudioModel = {
  model_type: string
  input_length: number
  feature_size: number
  frame_count: number
  bins_per_frame: number
  classes: number[]
  metrics: {
    dataset: string
    train_size: number
    validation_size: number
    test_size: number
    accuracy: number
    macro_f1: number
    input_length: number
    feature_size: number
    epochs_ran: number
  }
  weights: {
    dense1_kernel: number[][]
    dense1_bias: number[]
    dense2_kernel: number[][]
    dense2_bias: number[]
    dense3_kernel: number[][]
    dense3_bias: number[]
    out_kernel: number[][]
    out_bias: number[]
  }
}

const sampleDigits = Array.from({ length: 10 }, (_, digit) => ({
  label: `Digit ${digit}`,
  path: `/samples/digit_${digit}.wav`,
  digit,
}))

function relu(values: number[]) {
  return values.map((value) => Math.max(0, value))
}

function softmax(values: number[]) {
  const max = Math.max(...values)
  const exps = values.map((value) => Math.exp(value - max))
  const sum = exps.reduce((acc, value) => acc + value, 0)
  return exps.map((value) => value / sum)
}

function dense(input: number[], kernel: number[][], bias: number[]) {
  return bias.map((biasValue, outIndex) => {
    let sum = biasValue
    for (let inIndex = 0; inIndex < input.length; inIndex += 1) {
      sum += input[inIndex] * kernel[inIndex][outIndex]
    }
    return sum
  })
}

function resampleLinear(input: Float32Array, targetLength: number) {
  const output = new Array<number>(targetLength).fill(0)
  if (input.length === 0) return output

  for (let i = 0; i < targetLength; i += 1) {
    const position = (i / (targetLength - 1)) * (input.length - 1)
    const left = Math.floor(position)
    const right = Math.min(input.length - 1, left + 1)
    const weight = position - left
    output[i] = input[left] * (1 - weight) + input[right] * weight
  }

  const maxAbs = Math.max(...output.map((value) => Math.abs(value))) || 1
  return output.map((value) => value / maxAbs)
}

function bandEnergy(frame: number[], bandIndex: number, binsPerFrame: number) {
  const n = frame.length
  const start = Math.floor((bandIndex / binsPerFrame) * (n / 2))
  const end = Math.floor(((bandIndex + 1) / binsPerFrame) * (n / 2))
  let total = 0
  let count = 0

  for (let k = start; k < Math.max(end, start + 1); k += 1) {
    let real = 0
    let imag = 0
    for (let t = 0; t < n; t += 1) {
      const angle = (-2 * Math.PI * k * t) / n
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / Math.max(1, n - 1))
      real += frame[t] * window * Math.cos(angle)
      imag += frame[t] * window * Math.sin(angle)
    }
    total += Math.sqrt(real * real + imag * imag)
    count += 1
  }

  return Math.log1p(total / Math.max(1, count))
}

function extractFeatures(audio: number[], model: AudioModel) {
  const features: number[] = []
  const frameSize = Math.floor(audio.length / model.frame_count)

  for (let frameIndex = 0; frameIndex < model.frame_count; frameIndex += 1) {
    const frame = audio.slice(frameIndex * frameSize, (frameIndex + 1) * frameSize)
    for (let band = 0; band < model.bins_per_frame; band += 1) {
      features.push(bandEnergy(frame, band, model.bins_per_frame))
    }
  }

  const mean = features.reduce((sum, value) => sum + value, 0) / features.length
  const variance = features.reduce((sum, value) => sum + (value - mean) ** 2, 0) / features.length
  const std = Math.sqrt(variance) || 1
  return features.map((value) => (value - mean) / std)
}

function predict(features: number[], model: AudioModel) {
  const h1 = relu(dense(features, model.weights.dense1_kernel, model.weights.dense1_bias))
  const h2 = relu(dense(h1, model.weights.dense2_kernel, model.weights.dense2_bias))
  const h3 = relu(dense(h2, model.weights.dense3_kernel, model.weights.dense3_bias))
  return softmax(dense(h3, model.weights.out_kernel, model.weights.out_bias))
}

async function decodeAudio(fileOrUrl: Blob | File | string, inputLength: number) {
  const context = new AudioContext()
  const buffer =
    typeof fileOrUrl === "string"
      ? await fetch(fileOrUrl).then((response) => response.arrayBuffer())
      : await fileOrUrl.arrayBuffer()
  const decoded = await context.decodeAudioData(buffer.slice(0))
  const channel = decoded.getChannelData(0)
  await context.close()
  return resampleLinear(channel, inputLength)
}

export default function Home() {
  const [model, setModel] = useState<AudioModel | null>(null)
  const [audio, setAudio] = useState<number[] | null>(null)
  const [audioUrl, setAudioUrl] = useState("/samples/digit_5.wav")
  const [expected, setExpected] = useState(5)
  const [loadingAudio, setLoadingAudio] = useState(false)
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState("")
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    fetch("/model/spoken_digit_model.json")
      .then((response) => response.json())
      .then((data: AudioModel) => setModel(data))
  }, [])

  useEffect(() => {
    if (!model) return
    setLoadingAudio(true)
    decodeAudio(audioUrl, model.input_length)
      .then(setAudio)
      .catch((err) => setError(String(err)))
      .finally(() => setLoadingAudio(false))
  }, [audioUrl, model])

  const inference = useMemo(() => {
    if (!model || !audio) return null
    const t0 = performance.now()
    const features = extractFeatures(audio, model)
    const probabilities = predict(features, model)
    const latency = (performance.now() - t0).toFixed(1)
    return { features, probabilities, latency }
  }, [audio, model])

  const probabilities = inference?.probabilities ?? []
  const features = inference?.features ?? []
  const predicted = probabilities.length ? probabilities.indexOf(Math.max(...probabilities)) : -1
  const confidence = probabilities.length ? Math.max(...probabilities) : 0

  const waveformBars = useMemo(() => {
    if (!audio || audio.length === 0) return new Array<number>(64).fill(0.05)
    const count = 64
    const step = Math.floor(audio.length / count)
    return Array.from({ length: count }, (_, idx) => {
      const slice = audio.slice(idx * step, (idx + 1) * step)
      const peak = Math.max(...slice.map((v) => Math.abs(v)), 0.04)
      return Math.min(1, peak)
    })
  }, [audio])

  async function handleUpload(file: File | undefined) {
    if (!file || !model) return
    setLoadingAudio(true)
    setError("")
    try {
      const decoded = await decodeAudio(file, model.input_length)
      setAudio(decoded)
      setAudioUrl(URL.createObjectURL(file))
      setExpected(-1)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoadingAudio(false)
    }
  }

  async function startRecording() {
    if (!model) return
    setError("")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" })
        stream.getTracks().forEach((track) => track.stop())
        setLoadingAudio(true)
        try {
          const decoded = await decodeAudio(blob, model.input_length)
          setAudio(decoded)
          setAudioUrl(URL.createObjectURL(blob))
          setExpected(-1)
        } catch (err) {
          setError(`Could not decode recording: ${String(err)}`)
        } finally {
          setLoadingAudio(false)
        }
      }
      recorder.start()
      setRecording(true)
    } catch (err) {
      setError(`Microphone access failed: ${String(err)}`)
    }
  }

  function stopRecording() {
    recorderRef.current?.stop()
    setRecording(false)
  }

  return (
    <main className="min-h-screen bg-[#07050d] text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_18%_18%,rgba(168,85,247,0.14),transparent_38%),radial-gradient(circle_at_82%_75%,rgba(236,72,153,0.1),transparent_42%)]" />

      <header className="relative border-b border-purple-500/20 bg-[#0b0716]/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-purple-500/40 bg-gradient-to-br from-purple-600/30 to-fuchsia-600/20 text-purple-300">
              <AudioLines className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-black tracking-tight text-white">AcousticFFT Neural Lab</span>
                <span className="rounded-full border border-purple-500/30 bg-purple-950/60 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-purple-300">
                  16×32 Spectral Classifier
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Web Audio API Resampling · Hann-Windowed DFT Filterbank · 3-Layer Keras Dense Classifier
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 font-mono text-xs">
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-purple-500/30 bg-purple-950/40 px-3 py-1.5 text-purple-200">
              <Activity className="h-3.5 w-3.5 text-purple-400" />
              {inference ? `DSP + Forward Pass: ${inference.latency} ms` : "Initializing DSP..."}
            </span>
          </div>
        </div>
      </header>

      <section className="relative mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[400px_1fr]">
        {/* Left Acoustic Controls */}
        <aside className="space-y-5">
          <div className="rounded-2xl border border-purple-500/25 bg-[#0e091d]/90 p-5 shadow-2xl shadow-black/50">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-purple-300">
                Spoken Digit Reference Corpus (0–9)
              </span>
              <span className="font-mono text-[11px] text-slate-400">8 kHz WAV</span>
            </div>

            <div className="mb-4 grid grid-cols-5 gap-2">
              {sampleDigits.map((sample) => (
                <button
                  key={sample.digit}
                  onClick={() => {
                    setAudioUrl(sample.path)
                    setExpected(sample.digit)
                  }}
                  className={`rounded-xl border py-2.5 font-mono text-sm font-black transition ${
                    expected === sample.digit
                      ? "border-purple-400 bg-gradient-to-br from-purple-500 to-fuchsia-600 text-white shadow-lg shadow-purple-500/25"
                      : "border-purple-500/15 bg-[#080511] text-slate-300 hover:border-purple-500/40"
                  }`}
                >
                  {sample.digit}
                </button>
              ))}
            </div>

            <div className="grid gap-2.5">
              <button
                onClick={recording ? stopRecording : startRecording}
                className={`flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold transition ${
                  recording
                    ? "bg-rose-500 text-white shadow-lg shadow-rose-500/30"
                    : "bg-gradient-to-r from-purple-500 to-fuchsia-500 text-white shadow-lg shadow-purple-500/25 hover:from-purple-600 hover:to-fuchsia-600"
                }`}
              >
                {recording ? <CircleStop className="h-4 w-4 animate-pulse" /> : <Mic className="h-4 w-4" />}
                {recording ? "Stop Microphone Capture" : "Record Live Microphone Digit"}
              </button>

              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-purple-500/25 bg-[#080511] px-4 py-3 text-sm font-semibold text-slate-200 transition hover:border-purple-400">
                <Upload className="h-4 w-4 text-purple-400" />
                Upload Custom WAV / Audio File
                <input
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(event) => handleUpload(event.target.files?.[0])}
                />
              </label>
            </div>

            <audio src={audioUrl} controls className="mt-4 w-full rounded-lg" />

            {error && (
              <div className="mt-3 rounded-xl border border-rose-500/30 bg-rose-950/40 p-3 text-xs text-rose-200">
                {error}
              </div>
            )}

            {/* Prediction Readout Card */}
            <div className="mt-4 rounded-xl border border-purple-500/20 bg-[#080511] p-4">
              <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                <span>TOP SOFTMAX CLASS</span>
                <span>{expected >= 0 ? `Ground Truth: ${expected}` : "Custom Recording"}</span>
              </div>
              <div className="mt-2 flex items-baseline justify-between">
                <div className="font-mono text-5xl font-black text-white">
                  {loadingAudio ? "..." : predicted >= 0 ? predicted : "-"}
                </div>
                <div className="text-right">
                  <div className="font-mono text-xl font-black text-purple-300">
                    {(confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-slate-400">posterior confidence</div>
                </div>
              </div>
            </div>
          </div>

          {/* DSP Pipeline Spec */}
          <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-5">
            <div className="mb-3 flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-purple-300">
              <Waves className="h-4 w-4 text-purple-400" />
              Signal Processing Chain
            </div>
            <div className="space-y-2 font-mono text-xs text-slate-300">
              <div className="rounded-lg border border-purple-500/15 bg-[#080511] p-2.5">
                1. Decode PCM & linear resample → {model?.input_length ?? 4096} samples
              </div>
              <div className="rounded-lg border border-purple-500/15 bg-[#080511] p-2.5">
                2. Segment into {model?.frame_count ?? 16} frames × {model?.bins_per_frame ?? 32} Hann DFT log-energy bands
              </div>
              <div className="rounded-lg border border-purple-500/15 bg-[#080511] p-2.5">
                3. Per-utterance Z-score standardization (512 features)
              </div>
              <div className="rounded-lg border border-purple-500/15 bg-[#080511] p-2.5">
                4. Keras Dense(192) → Dense(96) → Dense(48) → Softmax(10)
              </div>
            </div>
          </div>
        </aside>

        {/* Right Visualizations */}
        <section className="space-y-6">
          {/* Top Metrics */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Corpus", model?.metrics.dataset || "Free Spoken Digit"],
              ["Test Accuracy", model ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : "..."],
              ["Macro F1 Score", model ? `${(model.metrics.macro_f1 * 100).toFixed(1)}%` : "..."],
              ["Feature Tensor", model ? `${model.frame_count}×${model.bins_per_frame} (${model.feature_size}D)` : "..."],
            ].map(([label, value]) => (
              <div key={label} className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-4">
                <div className="text-xs font-mono uppercase tracking-wider text-slate-400">{label}</div>
                <div className="mt-2 text-2xl font-black text-white">{value}</div>
              </div>
            ))}
          </div>

          {/* Waveform + Spectrogram Heatmap */}
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-6">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Volume2 className="h-5 w-5 text-purple-400" />
                  <h2 className="text-lg font-bold text-white">Normalized Time-Domain Waveform</h2>
                </div>
                <span className="font-mono text-xs text-slate-400">64-segment peak envelope</span>
              </div>

              <div className="flex h-44 items-center justify-between gap-1 rounded-xl border border-purple-500/15 bg-[#080511] px-4 py-3">
                {waveformBars.map((amp, i) => (
                  <div
                    key={i}
                    className="w-full rounded-full bg-gradient-to-t from-purple-600 via-fuchsia-500 to-purple-300 transition-all duration-200"
                    style={{ height: `${Math.max(8, Math.round(amp * 100))}%` }}
                  />
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-6">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-fuchsia-400" />
                  <h2 className="text-lg font-bold text-white">16×32 Log-DFT Spectrogram Tensor</h2>
                </div>
                <span className="font-mono text-xs text-slate-400">Z-scored input matrix</span>
              </div>

              <div
                className="grid h-44 gap-[2px] rounded-xl border border-purple-500/15 bg-[#080511] p-3"
                style={{ gridTemplateColumns: `repeat(32, minmax(0, 1fr))` }}
              >
                {(features.length ? features : new Array(512).fill(0)).slice(0, 512).map((val, idx) => {
                  const norm = Math.max(0, Math.min(1, (val + 1.5) / 3.5))
                  return (
                    <div
                      key={idx}
                      className="rounded-[2px]"
                      style={{
                        backgroundColor: `rgba(192, 132, 252, ${0.08 + norm * 0.88})`,
                      }}
                    />
                  )
                })}
              </div>
            </div>
          </div>

          {/* 10-Class Softmax Distribution */}
          <div className="rounded-2xl border border-purple-500/20 bg-[#0e091d]/90 p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-400" />
                <h2 className="text-lg font-bold text-white">10-Class Softmax Posterior Distribution</h2>
              </div>
              <span className="font-mono text-xs text-purple-300">
                {predicted >= 0 ? `Argmax = Digit ${predicted}` : "Waiting"}
              </span>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {Array.from({ length: 10 }, (_, digit) => {
                const prob = probabilities[digit] ?? 0
                const pct = (prob * 100).toFixed(1)
                const isWinner = digit === predicted
                return (
                  <div
                    key={digit}
                    className={`rounded-xl border p-3.5 transition ${
                      isWinner
                        ? "border-purple-400/60 bg-purple-950/40"
                        : "border-purple-500/15 bg-[#080511]"
                    }`}
                  >
                    <div className="mb-1.5 flex items-center justify-between font-mono text-xs">
                      <span className="font-bold text-white">Spoken Digit {digit}</span>
                      <span className={isWinner ? "font-black text-purple-300" : "text-slate-400"}>{pct}%</span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          isWinner ? "bg-gradient-to-r from-purple-500 to-fuchsia-400" : "bg-purple-500/40"
                        }`}
                        style={{ width: `${Math.max(2, prob * 100)}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}
