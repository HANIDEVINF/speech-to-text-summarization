"use client"

import { useEffect, useMemo, useState } from "react"
import { AudioLines, BarChart3, Brain, Database, FileAudio, Loader2, Mic, Play, Sparkles, Upload, Volume2, Wand2 } from "lucide-react"

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
  const sum = exps.reduce((total, value) => total + value, 0)
  return exps.map((value) => value / sum)
}

function dense(input: number[], kernel: number[][], bias: number[]) {
  return bias.map((biasValue, column) => {
    let sum = biasValue
    for (let row = 0; row < input.length; row += 1) {
      sum += input[row] * kernel[row][column]
    }
    return sum
  })
}

function resampleLinear(input: Float32Array, targetLength: number) {
  const output = new Array(targetLength).fill(0)
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

async function decodeAudio(fileOrUrl: File | string, inputLength: number) {
  const context = new AudioContext()
  const buffer = typeof fileOrUrl === "string" ? await fetch(fileOrUrl).then((response) => response.arrayBuffer()) : await fileOrUrl.arrayBuffer()
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
  const [error, setError] = useState("")

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

  const probabilities = useMemo(() => {
    if (!model || !audio) return []
    return predict(extractFeatures(audio, model), model)
  }, [audio, model])
  const predicted = probabilities.length ? probabilities.indexOf(Math.max(...probabilities)) : -1
  const confidence = probabilities.length ? Math.max(...probabilities) : 0

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

  return (
    <main className="min-h-screen bg-[#071015] text-slate-50">
      <section className="mx-auto grid min-h-screen max-w-7xl gap-8 px-6 py-10 lg:grid-cols-[390px_1fr]">
        <aside className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
          <div className="mb-6 flex items-center gap-3">
            <AudioLines className="h-9 w-9 text-cyan-300" />
            <div>
              <h1 className="text-2xl font-black">Keras Voice Model</h1>
              <p className="text-sm text-slate-400">Real spoken digit classifier trained on WAV audio.</p>
            </div>
          </div>

          <div className="mb-4 grid grid-cols-5 gap-2">
            {sampleDigits.map((sample) => (
              <button
                key={sample.digit}
                onClick={() => {
                  setAudioUrl(sample.path)
                  setExpected(sample.digit)
                }}
                className={`rounded-md border px-2 py-3 text-sm font-bold transition ${
                  expected === sample.digit ? "border-cyan-300 bg-cyan-300 text-slate-950" : "border-white/10 bg-slate-950"
                }`}
              >
                {sample.digit}
              </button>
            ))}
          </div>

          <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-white/10 bg-slate-950 px-4 py-3 text-sm font-bold transition hover:border-cyan-300/50">
            <Upload className="h-4 w-4" />
            Upload WAV Audio
            <input type="file" accept="audio/*" className="hidden" onChange={(event) => handleUpload(event.target.files?.[0])} />
          </label>

          <audio src={audioUrl} controls className="mt-4 w-full" />

          <div className="mt-4 rounded-md border border-white/10 bg-slate-950 p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
              <Play className="h-4 w-4" />
              Model result
            </div>
            <div className="text-5xl font-black text-cyan-200">{loadingAudio ? "..." : predicted >= 0 ? predicted : "-"}</div>
            <div className="mt-2 text-sm text-slate-400">
              {expected >= 0 ? `Expected sample label: ${expected}` : "Uploaded audio has no known label"}
            </div>
          </div>
        </aside>

        <section className="space-y-6">
          <div className="grid gap-4 md:grid-cols-4">
            {[
              ["Dataset", model?.metrics.dataset || "loading"],
              ["Test Accuracy", model ? `${(model.metrics.accuracy * 100).toFixed(1)}%` : "..."],
              ["Macro F1", model ? `${(model.metrics.macro_f1 * 100).toFixed(1)}%` : "..."],
              ["Samples", model ? String(model.metrics.train_size + model.metrics.validation_size + model.metrics.test_size) : "..."],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                <div className="text-sm text-slate-400">{label}</div>
                <div className="mt-2 text-2xl font-black text-cyan-200">{value}</div>
              </div>
            ))}
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
            <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
              <div className="mb-4 flex items-center gap-2">
                <Brain className="h-5 w-5 text-cyan-300" />
                <h2 className="text-xl font-bold">Keras Audio Inference</h2>
              </div>
              <div className="rounded-lg border border-white/10 bg-slate-950 p-5">
                <div className="mb-3 flex items-center gap-2 text-sm text-slate-400">
                  {loadingAudio ? <Loader2 className="h-4 w-4 animate-spin" /> : <Volume2 className="h-4 w-4" />}
                  Browser decodes audio, extracts FFT band-energy features, then runs exported Keras weights.
                </div>
                <div className="space-y-3">
                  {probabilities.map((probability, digit) => (
                    <div key={digit}>
                      <div className="mb-1 flex justify-between text-sm">
                        <span>Digit {digit}</span>
                        <span>{(probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/10">
                        <div className="h-2 rounded-full bg-cyan-300" style={{ width: `${probability * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
                {error && <p className="mt-4 text-sm text-red-200">{error}</p>}
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
                <div className="mb-4 flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-emerald-300" />
                  <h2 className="text-xl font-bold">Training Metrics</h2>
                </div>
                {model ? (
                  <div className="space-y-3 text-sm">
                    {[
                      ["Train", model.metrics.train_size],
                      ["Validation", model.metrics.validation_size],
                      ["Test", model.metrics.test_size],
                      ["Input length", model.metrics.input_length],
                      ["Feature size", model.metrics.feature_size],
                      ["Epochs", model.metrics.epochs_ran],
                    ].map(([label, value]) => (
                      <div key={label} className="flex justify-between rounded-md bg-slate-950 p-3">
                        <span className="text-slate-400">{label}</span>
                        <strong>{value}</strong>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-slate-400">Loading model...</div>
                )}
              </div>

              <div className="rounded-lg border border-white/10 bg-[#f5f2ea] p-5 text-slate-950">
                <div className="mb-3 flex items-center gap-2 font-black">
                  <Database className="h-5 w-5" />
                  Real DL Pipeline
                </div>
                <p className="text-sm leading-6 text-slate-700">
                  The repository includes dataset download, Keras training, saved model, exported browser weights,
                  sample WAV files, and metrics reports.
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-4 flex items-center gap-2">
              <Wand2 className="h-5 w-5 text-cyan-300" />
              <h2 className="text-xl font-bold">Audio Processing Trace</h2>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              {[
                ["Decode", "Web Audio API reads WAV/uploaded audio"],
                ["Resample", "Normalize to 8,000 waveform points"],
                ["Features", "16 frames x 32 log FFT bands"],
                ["Predict", `Dense neural net predicts digit ${predicted >= 0 ? predicted : ""}`],
              ].map(([title, copy]) => (
                <div key={title} className="rounded-md border border-white/10 bg-slate-950 p-4">
                  <FileAudio className="mb-3 h-5 w-5 text-cyan-200" />
                  <div className="font-bold">{title}</div>
                  <div className="mt-1 text-sm text-slate-400">{copy}</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}
