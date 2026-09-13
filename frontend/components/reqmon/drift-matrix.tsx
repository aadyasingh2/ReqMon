'use client'

import { useEffect, useState } from 'react'
import {
  AnimatePresence,
  animate,
  motion,
  useMotionValue,
  useTransform,
} from 'framer-motion'
import { Play } from 'lucide-react'

interface ModelData {
  observed: number
  target: number
  op: string
  businessReq: 'PASS' | 'FAIL'
  drift: 'STABLE' | 'DRIFT DETECTED'
  narrative: string[]
  features: { name: string; psi: number }[]
}

const MODEL_SOURCES: Record<string, string> = {
  RANDOM_FOREST: 'local:models/baseline_classifier.pkl',
  LOGISTIC_REG: 'local:models/alt_classifier_logreg.pkl',
  REMOTE_NODE: 'remote:http://localhost:5000',
}

const TABS = Object.keys(MODEL_SOURCES)

const DEFAULT_FEATURES = [
  { name: 'LOCATION_DISTANCE_KM', psi: 0.0 },
  { name: 'TRANSACTION_AMOUNT', psi: 0.0 },
  { name: 'DEVICE_RISK_SCORE', psi: 0.0 },
  { name: 'FAILED_PASSWORDS', psi: 0.0 },
]

export function DriftMatrix({
  requirementText,
}: {
  requirementText?: string
}) {
  const [tab, setTab] = useState(TABS[0])
  const [dataSource, setDataSource] = useState<'clean' | 'drifted'>('clean')
  const [data, setData] = useState<ModelData | null>(null)
  const [ran, setRan] = useState(false)
  const [loading, setLoading] = useState(false)

  async function run() {
    setLoading(true)
    setRan(false)
    try {
      const res = await fetch('http://localhost:8000/monitor/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requirement_text: requirementText ?? 'Recall must remain above 93%',
          model_source: MODEL_SOURCES[tab],
          data_source: dataSource,
        }),
      })
      const json = await res.json()
      setData({
        observed: json.observed_value,
        target: json.business_requirement_threshold,
        op: json.operator ?? '≥',
        businessReq: json.business_requirement_violation ? 'FAIL' : 'PASS',
        drift: json.operational_drift_violation ? 'DRIFT DETECTED' : 'STABLE',
        narrative: json.explanation.split('. ').filter(Boolean),
        features: json.contributors.map((c: any) => ({
          name: c.feature.toUpperCase(),
          psi: c.psi,
        })),
      })
      setRan(true)
    } catch (err) {
      console.error('Monitor check failed', err)
    } finally {
      setLoading(false)
    }
  }

  function selectTab(t: string) {
    setTab(t)
    setRan(false)
    setLoading(false)
    setData(null)
  }

  return (
    <section className="relative overflow-hidden rounded-lg border-x border-b border-white/5 border-t border-t-white/10 bg-surface">
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
            MODULE_02
          </span>
          <h2 className="font-sans text-sm font-semibold tracking-tight text-zinc-100">
            Drift &amp; Inference Matrix
          </h2>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
          X:02 Y:00
        </span>
      </div>

      <div className="flex flex-col gap-4 border-b border-white/[0.06] px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <div
            className="flex gap-1 rounded-md border border-white/10 bg-zinc-950 p-1"
            role="tablist"
            aria-label="Model selector"
          >
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={t === tab}
                onClick={() => selectTab(t)}
                className={`relative whitespace-nowrap rounded px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40 ${
                  t === tab ? 'text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {t === tab && (
                  <motion.span
                    layoutId="activeTab"
                    transition={{ type: 'spring', stiffness: 320, damping: 28 }}
                    className="absolute inset-0 rounded border border-white/10 bg-zinc-800 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                    aria-hidden="true"
                  />
                )}
                <span className="relative z-10 flex items-center gap-1.5">
                  {t === tab && (
                    <span className="h-1 w-1 rounded-full bg-indigo-400 shadow-[0_0_6px_rgba(129,140,248,0.9)]" />
                  )}
                  {t}
                </span>
              </button>
            ))}
          </div>

          <div
            className="flex gap-1 rounded-md border border-white/10 bg-zinc-950 p-1"
            role="group"
            aria-label="Dataset selector"
          >
            {[
              { key: 'clean', label: 'PRODUCTION_CLEAN' },
              { key: 'drifted', label: 'PRODUCTION_DRIFTED' },
            ].map((ds) => (
              <button
                key={ds.key}
                type="button"
                onClick={() => {
                  setDataSource(ds.key as 'clean' | 'drifted')
                  setRan(false)
                  setData(null)
                }}
                className={`relative whitespace-nowrap rounded px-3 py-1.5 font-mono text-[10px] uppercase tracking-widest transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40 ${
                  dataSource === ds.key ? 'text-indigo-300' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {dataSource === ds.key && (
                  <motion.span
                    layoutId="activeDataSource"
                    transition={{ type: 'spring', stiffness: 320, damping: 28 }}
                    className="absolute inset-0 rounded border border-indigo-500/30 bg-indigo-950/40 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                    aria-hidden="true"
                  />
                )}
                <span className="relative z-10 flex items-center gap-1.5">
                  {dataSource === ds.key && (
                    <span className="h-1 w-1 rounded-full bg-indigo-400 shadow-[0_0_6px_rgba(129,140,248,0.9)]" />
                  )}
                  {ds.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        <motion.button
          type="button"
          onClick={run}
          whileTap={{ scale: 0.98 }}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-white/10 bg-zinc-800 px-4 py-2 font-mono text-xs font-medium uppercase tracking-widest text-zinc-200 transition-all hover:bg-zinc-700 hover:shadow-[0_0_12px_rgba(255,255,255,0.05)] focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/40"
        >
          <Play className="h-3 w-3 fill-current" />
          Run_Check
        </motion.button>
      </div>

      <div className="grid grid-cols-1 gap-px bg-white/[0.06] lg:grid-cols-2">
        <div className="bg-surface p-4">
          <LeftColumn data={data} ran={ran} loading={loading} />
        </div>
        <div className="bg-surface p-4">
          <RightColumn data={data} ran={ran} loading={loading} />
        </div>
      </div>
    </section>
  )
}

function LeftColumn({
  data,
  ran,
  loading,
}: {
  data: ModelData | null
  ran: boolean
  loading: boolean
}) {
  const fail = data?.businessReq === 'FAIL'
  return (
    <div className="space-y-4">
      <ObservedGauge
        value={data?.observed ?? 0}
        target={data?.target ?? 0.93}
        op={data?.op ?? '≥'}
        fail={fail}
        ran={ran}
        loading={loading}
      />

      <div className="grid grid-cols-2 gap-3">
        <StatusCard label="Business_Req" ran={ran} loading={loading}>
          <span
            className={`inline-flex items-center rounded border px-2.5 py-1 font-mono text-xs font-medium ${
              fail
                ? 'border-rose-500/40 bg-rose-950/30 text-rose-400 shadow-[0_0_16px_-6px_rgba(244,63,94,0.6)]'
                : 'border-emerald-500/40 bg-emerald-950/30 text-emerald-400 shadow-[0_0_16px_-6px_rgba(16,185,129,0.6)]'
            }`}
          >
            {data?.businessReq ?? '—'}
          </span>
        </StatusCard>
        <StatusCard label="Operational_Drift" ran={ran} loading={loading}>
          <span
            className={`inline-flex items-center rounded border px-2.5 py-1 font-mono text-xs font-medium ${
              data?.drift === 'DRIFT DETECTED'
                ? 'border-amber-500/40 bg-amber-950/30 text-amber-400'
                : 'border-emerald-500/40 bg-emerald-950/30 text-emerald-400'
            }`}
          >
            {data?.drift ? data.drift.replace(' ', '_') : '—'}
          </span>
        </StatusCard>
      </div>

      <div className="rounded-md border border-white/[0.06] bg-surface-3">
        <div className="border-b border-white/[0.06] px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          Diagnostic_Narrative
        </div>
        <div className="space-y-2 p-4">
          {loading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="reqmon-skeleton h-3 rounded"
                style={{ width: `${90 - i * 8}%` }}
              />
            ))
          ) : ran && data ? (
            <motion.div
              className="space-y-2"
              initial="hidden"
              animate="show"
              variants={{
                hidden: {},
                show: { transition: { staggerChildren: 0.06 } },
              }}
            >
              {data.narrative.map((line, i) => (
                <motion.p
                  key={i}
                  variants={{
                    hidden: { opacity: 0, x: -8 },
                    show: { opacity: 1, x: 0 },
                  }}
                  className="font-mono text-xs leading-relaxed text-zinc-400"
                >
                  <span className="mr-2 text-zinc-600">
                    {(i + 1).toString().padStart(2, '0')}
                  </span>
                  {line}
                </motion.p>
              ))}
            </motion.div>
          ) : (
            <p className="font-mono text-xs text-zinc-600">
              Run a monitoring check to generate a diagnostic narrative.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function ObservedGauge({
  value,
  target,
  op,
  fail,
  ran,
  loading,
}: {
  value: number
  target: number
  op: string
  fail: boolean
  ran: boolean
  loading: boolean
}) {
  const R = 35
  const C = 2 * Math.PI * R
  const color = fail ? '#f43f5e' : '#10b981'

  const progress = useMotionValue(0)
  const dashOffset = useTransform(progress, (p) => C - C * Math.min(p, 1))
  const display = useTransform(progress, (p) => p.toFixed(4))

  useEffect(() => {
    if (ran) {
      const controls = animate(progress, value, {
        duration: 0.6,
        ease: [0.22, 1, 0.36, 1],
      })
      return () => controls.stop()
    }
    progress.set(0)
  }, [ran, value, progress])

  return (
    <div className="flex items-center gap-5 rounded-md border border-white/[0.06] bg-white/[0.02] p-4">
      <div className="relative h-24 w-24 shrink-0">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 80 80">
          <circle
            cx="40"
            cy="40"
            r={R}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="2"
          />
          <motion.circle
            cx="40"
            cy="40"
            r={R}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={ran ? dashOffset : C}
            style={{ filter: `drop-shadow(0 0 4px ${color}66)` }}
          />
        </svg>
        {ran && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: color, boxShadow: `0 0 8px ${color}` }}
            />
          </div>
        )}
      </div>
      <div>
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          Observed_Value
        </div>
        {loading ? (
          <div className="reqmon-skeleton mt-2 h-9 w-28 rounded" />
        ) : ran ? (
          <motion.div
            className="mt-1 font-mono text-4xl font-semibold tabular-nums tracking-tight"
            style={{ color: fail ? '#f43f5e' : '#fafafa' }}
          >
            {display}
          </motion.div>
        ) : (
          <div className="mt-1 font-mono text-4xl font-semibold tabular-nums tracking-tight text-zinc-100">
            —
          </div>
        )}
        <div className="mt-1 font-mono text-[10px] uppercase tracking-widest text-zinc-600">
          Target {op || '≥'} {target ? target.toFixed(4) : '0.9300'}
        </div>
      </div>
    </div>
  )
}

function StatusCard({
  label,
  children,
  ran,
  loading,
}: {
  label: string
  children: React.ReactNode
  ran: boolean
  loading: boolean
}) {
  return (
    <div className="rounded-md border border-white/[0.06] bg-white/[0.02] p-4">
      <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
        {label}
      </div>
      <div className="mt-2.5">
        {loading ? (
          <div className="reqmon-skeleton h-6 w-24 rounded" />
        ) : ran ? (
          <motion.span
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ type: 'spring', stiffness: 260, damping: 20 }}
            className="inline-block"
          >
            {children}
          </motion.span>
        ) : (
          <span className="font-mono text-sm text-zinc-600">—</span>
        )}
      </div>
    </div>
  )
}

function RightColumn({
  data,
  ran,
  loading,
}: {
  data: ModelData | null
  ran: boolean
  loading: boolean
}) {
  const features = data?.features ?? DEFAULT_FEATURES

  return (
    <div>
      <div className="mb-4 font-mono text-[10px] uppercase tracking-widest text-zinc-500">
        // Drift_Contributor_Ranking
      </div>
      <motion.div
        className="space-y-4"
        key={ran ? 'ran' : 'idle'}
        initial="hidden"
        animate="show"
        variants={{ hidden: {}, show: { transition: { staggerChildren: 0.06 } } }}
      >
        {features.map((f) => {
          const hot = f.psi > 0.25
          return (
            <motion.div
              key={f.name}
              variants={{
                hidden: { opacity: 0, y: 8 },
                show: { opacity: 1, y: 0 },
              }}
              className="space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-zinc-300">{f.name}</span>
                {loading ? (
                  <div className="reqmon-skeleton h-3.5 w-12 rounded" />
                ) : (
                  <span
                    className="font-mono text-xs tabular-nums"
                    style={{ color: ran ? (hot ? '#fb7185' : '#94a3b8') : '#52525b' }}
                  >
                    {ran ? f.psi.toFixed(4) : '0.0000'}
                  </span>
                )}
              </div>
              <div className="h-0.5 overflow-hidden rounded-full bg-zinc-800">
                <motion.div
                  className="h-full rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: ran ? `${Math.min(f.psi / 0.5, 1) * 100}%` : '0%' }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  style={{
                    background: hot
                      ? 'linear-gradient(90deg, rgba(244,63,94,0.4), rgba(244,63,94,0.8))'
                      : 'linear-gradient(90deg, rgba(51,65,85,0.6), rgba(100,116,139,0.8))',
                  }}
                />
              </div>
            </motion.div>
          )
        })}
      </motion.div>
      <div className="mt-4 flex items-center justify-between border-t border-white/[0.06] pt-3 font-mono text-[10px] uppercase tracking-widest text-zinc-600">
        <span>PSI &gt; 0.25 = CRITICAL</span>
        <span className="tabular-nums">n={features.length}</span>
      </div>
    </div>
  )
}
