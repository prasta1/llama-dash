import { useEffect, useRef, useState } from 'react'
import { Sparkline } from '../../components/Sparkline'

/**
 * `tone` is the *state* the value is reporting, not the identity of the metric.
 * The error-rate card previously passed a static `color="var(--err)"`, so it
 * drew red at 0.0% — the loudest element on the dashboard reporting that
 * nothing was wrong.
 */
type StatTone = 'neutral' | 'err'

type Props = {
  label: string
  value: string
  unit: string
  sparkline?: Array<number>
  tone?: StatTone
}

const TONE_COLOR: Record<StatTone, string> = {
  neutral: 'var(--accent)',
  err: 'var(--err)',
}

export function DashboardStatCard({ label, value, unit, sparkline, tone = 'neutral' }: Props) {
  const animated = useCountUp(value)
  return (
    <div className="stat-card group relative min-h-[68px] overflow-hidden transition-[background-color] duration-150 hover:bg-[color-mix(in_srgb,var(--accent)_4%,var(--bg-1)_88%,var(--bg-2))]">
      <div className="stat-card-label">{label}</div>
      <div className="stat-card-row">
        <span className={tone === 'err' ? 'stat-card-value text-err' : 'stat-card-value'}>{animated}</span>
        <span className="stat-card-unit">{unit}</span>
      </div>
      {sparkline ? (
        <div className="opacity-90">
          <Sparkline data={sparkline} height={32} color={TONE_COLOR[tone]} showEndpointDot />
        </div>
      ) : null}
    </div>
  )
}

/**
 * Smoothly interpolates between the previous and next numeric string value.
 * Parses the leading number, animates it, and re-attaches the suffix. Falls
 * back to the raw value when no number is found (e.g. "—").
 */
function useCountUp(value: string, duration = 450): string {
  const [display, setDisplay] = useState(value)
  const fromRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const match = value.match(/^(-?[\d.]+)(.*)$/)
    if (!match) {
      setDisplay(value)
      fromRef.current = null
      return
    }
    const target = Number.parseFloat(match[1])
    const suffix = match[2]
    if (!Number.isFinite(target)) {
      setDisplay(value)
      return
    }
    const from = fromRef.current ?? target
    if (from === target) {
      setDisplay(value)
      return
    }
    const start = performance.now()
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - t, 3)
      const current = from + (target - from) * eased
      const decimals = match[1].includes('.') ? match[1].split('.')[1].length : 0
      setDisplay(`${current.toFixed(decimals)}${suffix}`)
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = target
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current)
      fromRef.current = target
    }
  }, [value, duration])

  return display
}
