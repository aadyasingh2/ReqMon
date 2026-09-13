'use client'

import { useEffect, useState } from 'react'
import { Volume2, VolumeX } from 'lucide-react'

export function SystemHeader() {
  const [audio, setAudio] = useState(true)
  const [latency, setLatency] = useState(12)

  useEffect(() => {
    const id = setInterval(() => {
      setLatency(9 + Math.floor(Math.random() * 8))
    }, 2000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="flex flex-col gap-4 border-b border-white/[0.06] px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
      <div className="flex items-center gap-3">
        <span className="font-mono text-base font-bold tracking-tight text-zinc-100">
          REQMON
        </span>
        <span className="flex items-center gap-2">
          <span
            className="h-1.5 w-1.5 rounded-full bg-emerald"
            style={{ animation: 'pulse-dot 2s ease-in-out infinite' }}
            aria-hidden="true"
          />
          <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            // DRIFT_ENGINE_ONLINE
          </span>
        </span>
      </div>

      <div className="flex items-center gap-2.5">
        <button
          type="button"
          onClick={() => setAudio((a) => !a)}
          className="flex items-center gap-2 rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest text-zinc-400 transition-colors hover:border-white/20 hover:bg-zinc-800/60 hover:text-zinc-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40"
          aria-pressed={audio}
        >
          {audio ? (
            <Volume2 className="h-3.5 w-3.5 text-emerald" />
          ) : (
            <VolumeX className="h-3.5 w-3.5 text-zinc-500" />
          )}
          AUDIO_HAPTIC: {audio ? 'ACTIVE' : 'MUTED'}
        </button>

        <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-1.5">
          <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            LAT
          </span>
          <span className="font-mono text-xs tabular-nums text-zinc-200">
            {latency}ms
          </span>
        </div>
      </div>
    </header>
  )
}
