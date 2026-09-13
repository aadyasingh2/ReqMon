'use client'

import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'

interface ParseResult {
  ambiguous: boolean
  metric: string
  op: string
  thresh: string
  sev: string
  ambiguityReason: string | null
}

export function RequirementParser({
  value: externalValue,
  onChange: externalOnChange,
}: {
  value?: string
  onChange?: (val: string) => void
}) {
  const [internalValue, setInternalValue] = useState('Recall must remain above 93%')
  const value = externalValue !== undefined ? externalValue : internalValue
  const setValue = externalOnChange ?? setInternalValue

  const [result, setResult] = useState<ParseResult | null>(null)
  const [loading, setLoading] = useState(false)

  async function interpret() {
    const next = value
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch('http://localhost:8000/interpret', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: next }),
      })
      const data = await res.json()
      setResult({
        ambiguous: data.is_ambiguous,
        metric: data.metric?.toUpperCase() ?? 'UNKNOWN',
        op: data.operator ?? '',
        thresh: data.threshold != null ? data.threshold.toFixed(4) : '',
        sev: data.severity?.toUpperCase() ?? 'STANDARD',
        ambiguityReason: data.ambiguity_reason ?? null,
      })
    } catch (err) {
      console.error('Interpret request failed', err)
      setResult({
        ambiguous: true,
        metric: 'UNKNOWN',
        op: '',
        thresh: '',
        sev: 'STANDARD',
        ambiguityReason: 'Failed to connect to monitoring backend service.',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="relative overflow-hidden rounded-lg border-x border-b border-white/5 border-t border-t-white/10 bg-surface">
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            MODULE_01
          </span>
          <h2 className="font-sans text-sm font-semibold tracking-tight text-zinc-100">
            Requirement Parser
          </h2>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
          X:01 Y:00
        </span>
      </div>

      <div className="space-y-4 px-4 py-4">
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="flex flex-1 items-center gap-2.5 rounded-md border border-l-2 border-white/10 border-l-indigo-500 bg-zinc-950 px-3.5 py-2.5 transition-colors focus-within:border-l-indigo-400">
            <span className="select-none font-mono text-xs text-indigo-400">REQ_IN&gt;</span>
            <input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === 'Enter' &&
                  !e.nativeEvent.isComposing &&
                  e.keyCode !== 229
                ) {
                  interpret()
                }
              }}
              spellCheck={false}
              placeholder="Describe a requirement in plain language…"
              className="w-full bg-transparent font-mono text-sm text-zinc-200 outline-none placeholder:text-zinc-600"
              aria-label="Requirement input"
            />
          </div>

          <motion.button
            type="button"
            onClick={interpret}
            whileTap={{ scale: 0.98 }}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-md border border-white/10 bg-zinc-800 px-4 py-2.5 font-mono text-xs font-medium uppercase tracking-widest text-zinc-200 transition-all hover:bg-zinc-700 hover:shadow-[0_0_12px_rgba(255,255,255,0.05)] focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40"
          >
            Interpret
          </motion.button>
        </div>

        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-white/[0.06] bg-white/[0.06] sm:grid-cols-4"
            >
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="reqmon-skeleton h-[70px] bg-surface" />
              ))}
            </motion.div>
          ) : result ? (
            result.ambiguous ? (
              <motion.div
                key="ambiguous"
                initial={{ opacity: 0, y: 12, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ type: 'spring', stiffness: 260, damping: 20 }}
                className="flex items-start gap-3 rounded-md border border-l-2 border-rose-500/40 border-l-rose-500 bg-rose-950/30 px-4 py-3.5"
              >
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                <div>
                  <p className="font-mono text-xs font-medium uppercase tracking-widest text-rose-400">
                    ERR_014 // AMBIGUOUS_METRIC_BOUNDS
                  </p>
                  <p className="mt-1 font-mono text-xs leading-relaxed text-zinc-400">
                    {result.ambiguityReason ??
                      'Define a metric, a comparison operator, and an explicit numeric threshold value.'}
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 12, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ type: 'spring', stiffness: 260, damping: 20 }}
                className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-white/[0.06] bg-white/[0.06] sm:grid-cols-4"
              >
                <MetricPill label="Metric" value={result.metric} />
                <MetricPill label="Operator" value={result.op} />
                <MetricPill label="Threshold" value={result.thresh} />
                <MetricPill
                  label="Severity"
                  value={result.sev}
                  critical={result.sev === 'CRITICAL'}
                />
              </motion.div>
            )
          ) : (
            <p className="font-mono text-xs text-zinc-600">
              Enter a requirement and interpret it to extract a structured
              monitoring rule.
            </p>
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}

function MetricPill({
  label,
  value,
  critical,
}: {
  label: string
  value: string
  critical?: boolean
}) {
  return (
    <div className="bg-surface px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
        {label}
      </div>
      {critical !== undefined ? (
        <span
          className={`mt-2 inline-flex items-center rounded border px-2 py-0.5 font-mono text-xs font-medium ${
            critical
              ? 'border-rose-500/40 bg-rose-950/30 text-rose-400'
              : 'border-white/15 bg-white/5 text-zinc-300'
          }`}
        >
          {value}
        </span>
      ) : (
        <div className="mt-1.5 font-mono text-lg tabular-nums text-zinc-100">
          {value || '—'}
        </div>
      )}
    </div>
  )
}
