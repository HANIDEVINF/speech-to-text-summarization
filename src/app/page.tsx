"use client"

import { useMemo, useState } from "react"
import { AudioLines, BookOpen, CheckCircle2, Clock, FileText, Mic, Play, Search, Sparkles, UserRound, Wand2 } from "lucide-react"

type SampleKey = "lecture" | "sales" | "clinical"

const samples = {
  lecture: {
    label: "AI Lecture",
    duration: "38:12",
    speakers: ["Professor", "Student"],
    transcript:
      "Professor: Today we introduce retrieval augmented generation. The main idea is to retrieve relevant context before asking the language model to answer. Student: Why does that reduce hallucinations? Professor: Because the model is grounded in documents instead of only relying on memorized knowledge. The pipeline usually includes chunking, embeddings, vector search, reranking, and answer generation with citations.",
  },
  sales: {
    label: "Sales Call",
    duration: "22:48",
    speakers: ["Account Executive", "Customer"],
    transcript:
      "Account Executive: You mentioned the team loses time creating weekly reports. Customer: Yes, around six hours each week. Account Executive: The automation plan would connect your CRM, summarize pipeline movement, and notify managers when risk is detected. Customer: Please send a security overview and pricing for twenty seats.",
  },
  clinical: {
    label: "Clinic Note",
    duration: "11:03",
    speakers: ["Clinician", "Patient"],
    transcript:
      "Clinician: The patient reports mild cough and fatigue for four days with no chest pain. Patient: I have been sleeping poorly. Clinician: We discussed hydration, rest, monitoring symptoms, and follow up if fever appears or breathing becomes difficult. Current medication list was reviewed and no new allergies were reported.",
  },
} satisfies Record<SampleKey, { label: string; duration: string; speakers: string[]; transcript: string }>

function summarize(transcript: string) {
  const lower = transcript.toLowerCase()
  const keywords = ["retrieval", "grounded", "automation", "security", "pricing", "hydration", "follow up", "symptoms"].filter((word) =>
    lower.includes(word),
  )
  const sentences = transcript
    .replaceAll("Professor:", "")
    .replaceAll("Student:", "")
    .replaceAll("Account Executive:", "")
    .replaceAll("Customer:", "")
    .replaceAll("Clinician:", "")
    .replaceAll("Patient:", "")
    .split(/[.!?]/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)

  return {
    short: sentences.slice(0, 2).join(". ") + ".",
    bullets: sentences.slice(0, 5),
    actions: [
      lower.includes("pricing") ? "Send pricing for 20 seats" : "Create study notes from transcript",
      lower.includes("security") ? "Share security overview" : "Tag key discussion moments",
      lower.includes("follow up") ? "Schedule follow-up if symptoms worsen" : "Store summary in searchable notes",
    ],
    keywords,
    confidence: Math.min(97, 78 + keywords.length * 3),
  }
}

export default function Home() {
  const [sample, setSample] = useState<SampleKey>("lecture")
  const [query, setQuery] = useState("What are the key decisions and action items?")
  const [processing, setProcessing] = useState(false)
  const [completedRuns, setCompletedRuns] = useState(7)
  const active = samples[sample]
  const summary = useMemo(() => summarize(active.transcript), [active.transcript, completedRuns])
  const chunks = active.transcript.split(". ").filter(Boolean)

  function runPipeline() {
    setProcessing(true)
    window.setTimeout(() => {
      setProcessing(false)
      setCompletedRuns((value) => value + 1)
    }, 650)
  }

  return (
    <main className="min-h-screen bg-[#071015] text-slate-50">
      <section className="mx-auto grid min-h-screen max-w-7xl gap-8 px-6 py-10 lg:grid-cols-[370px_1fr]">
        <aside className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-cyan-300 text-slate-950">
              <AudioLines className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-2xl font-black">EchoBrief</h1>
              <p className="text-sm text-slate-400">Speech-to-text summarization pipeline</p>
            </div>
          </div>

          <div className="space-y-2">
            {(Object.keys(samples) as SampleKey[]).map((key) => (
              <button
                key={key}
                onClick={() => setSample(key)}
                className={`w-full rounded-md border p-3 text-left transition ${
                  sample === key ? "border-cyan-300 bg-cyan-300/10" : "border-white/10 bg-slate-950"
                }`}
              >
                <span className="block font-bold">{samples[key].label}</span>
                <span className="text-sm text-slate-400">{samples[key].duration} · {samples[key].speakers.length} speakers</span>
              </button>
            ))}
          </div>

          <div className="mt-5 rounded-md border border-white/10 bg-slate-950 p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
              <Search className="h-4 w-4" />
              Ask the transcript
            </div>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="h-24 w-full resize-none rounded-md border border-white/10 bg-[#071015] p-3 text-sm outline-none focus:border-cyan-300"
            />
          </div>

          <button
            onClick={runPipeline}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-md bg-cyan-300 px-4 py-3 font-black text-slate-950"
          >
            {processing ? <Sparkles className="h-4 w-4 animate-pulse" /> : <Play className="h-4 w-4" />}
            {processing ? "Processing audio..." : "Run Pipeline"}
          </button>
        </aside>

        <section className="space-y-6">
          <div className="grid gap-4 md:grid-cols-4">
            {[
              ["Duration", active.duration],
              ["Speakers", String(active.speakers.length)],
              ["Confidence", `${summary.confidence}%`],
              ["Runs", String(completedRuns)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                <div className="text-sm text-slate-400">{label}</div>
                <div className="mt-2 text-2xl font-black text-cyan-200">{value}</div>
              </div>
            ))}
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
            <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
              <div className="mb-4 flex items-center gap-2">
                <Mic className="h-5 w-5 text-cyan-300" />
                <h2 className="text-xl font-bold">Live Transcript</h2>
              </div>
              <div className="space-y-3">
                {chunks.map((chunk, index) => (
                  <div key={chunk} className="rounded-md border border-white/10 bg-slate-950 p-4">
                    <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
                      <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> {String(index * 42).padStart(2, "0")}s</span>
                      <span>{index % 2 === 0 ? active.speakers[0] : active.speakers[1] || active.speakers[0]}</span>
                    </div>
                    <p className="leading-7 text-slate-200">{chunk}.</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
                <div className="mb-4 flex items-center gap-2">
                  <Wand2 className="h-5 w-5 text-emerald-300" />
                  <h2 className="text-xl font-bold">AI Summary</h2>
                </div>
                <p className="leading-7 text-slate-200">{summary.short}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {summary.keywords.map((keyword) => (
                    <span key={keyword} className="rounded-full bg-emerald-300/10 px-3 py-1 text-xs text-emerald-100">
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
                <div className="mb-4 flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-amber-300" />
                  <h2 className="text-xl font-bold">Action Items</h2>
                </div>
                <div className="space-y-3">
                  {summary.actions.map((action) => (
                    <div key={action} className="rounded-md bg-slate-950 p-3 text-sm text-slate-200">
                      {action}
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-white/10 bg-[#f5f2ea] p-5 text-slate-950">
                <div className="mb-3 flex items-center gap-2 font-black">
                  <BookOpen className="h-5 w-5" />
                  Transcript Q&A
                </div>
                <p className="text-sm leading-6 text-slate-700">
                  Query: {query} The assistant answers using the transcript, detected speakers, and generated summary.
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-4 flex items-center gap-2">
              <FileText className="h-5 w-5 text-cyan-300" />
              <h2 className="text-xl font-bold">Pipeline Trace</h2>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              {["Audio chunking", "Speech recognition", "Speaker diarization", "LLM summary"].map((step, index) => (
                <div key={step} className="rounded-md border border-white/10 bg-slate-950 p-4">
                  <UserRound className="mb-3 h-5 w-5 text-cyan-200" />
                  <div className="font-bold">{index + 1}. {step}</div>
                  <div className="mt-1 text-xs text-slate-500">cached · auditable · secure</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}
