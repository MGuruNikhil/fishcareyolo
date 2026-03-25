import { useState } from "react"
import { ChevronDown, ChevronUp, AlertCircle, AlertTriangle, CheckCircle } from "lucide-react"
import type { Detection } from "@/lib/model/types"
import { DISEASE_INFO } from "@/lib/model/disease"

interface Props {
  detection: Detection
}

export function DetectionCard({ detection }: Props) {
  const [expanded, setExpanded] = useState(false)
  const info = DISEASE_INFO[detection.diseaseClass]

  const severityColor = (() => {
    switch (info.severity) {
      case "healthy":
        return "var(--healthy)"
      case "low":
        return "var(--low)"
      case "medium":
      case "high":
        return "var(--medium)"
    }
  })()

  const severityBg = (() => {
    switch (info.severity) {
      case "healthy":
        return "var(--healthy-bg)"
      case "low":
        return "var(--low-bg)"
      case "medium":
      case "high":
        return "var(--medium-bg)"
    }
  })()

  const severityBorder = (() => {
    switch (info.severity) {
      case "healthy":
        return "var(--healthy-border)"
      case "low":
        return "var(--low-border)"
      case "medium":
      case "high":
        return "var(--medium-border)"
    }
  })()

  const Icon = (() => {
    switch (info.severity) {
      case "healthy":
        return CheckCircle
      case "low":
        return AlertTriangle
      case "medium":
      case "high":
        return AlertCircle
    }
  })()

  // For healthy, show "Healthy" instead of "healthy severity"
  const severityLabel = info.severity === "healthy" ? "Healthy" : `${info.severity} severity`

  return (
    <article
      className="overflow-hidden rounded-md border bg-card"
      style={{ borderLeftWidth: "3px", borderLeftColor: severityColor }}
    >
      <button
        className="flex w-full min-h-14 items-center justify-between px-4 py-4 text-left transition-colors hover:bg-muted/50"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-controls={`det-body-${detection.id}`}
      >
        <div className="flex flex-1 min-w-0 items-start gap-3">
          <Icon size={15} style={{ color: severityColor }} aria-hidden="true" />
          <div className="min-w-0 flex-1">
            <p className="mb-1 overflow-hidden text-ellipsis whitespace-nowrap text-sm font-semibold text-foreground">
              {info.displayName}
            </p>
            <span
              className="inline-flex items-center rounded-full border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.04em]"
              style={{
                backgroundColor: severityBg,
                color: severityColor,
                borderColor: severityBorder,
              }}
            >
              {severityLabel}
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2 text-secondary-foreground">
          <span
            className="font-mono text-sm font-semibold text-foreground"
            aria-label={`${(detection.confidence * 100).toFixed(0)}% confidence`}
          >
            {(detection.confidence * 100).toFixed(0)}%
          </span>
          {expanded ? (
            <ChevronUp size={14} aria-hidden="true" />
          ) : (
            <ChevronDown size={14} aria-hidden="true" />
          )}
        </div>
      </button>

      {expanded && (
        <div className="flex flex-col gap-4 border-t px-4 pb-4" id={`det-body-${detection.id}`}>
          <p className="pt-3 text-sm leading-relaxed text-secondary-foreground">
            {info.description}
          </p>

          {info.symptoms.length > 0 && (
            <div className="flex flex-col gap-2">
              <h4 className="font-mono text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                Symptoms
              </h4>
              <ul className="flex flex-col gap-1 pl-5">
                {info.symptoms.map((s: string, i: number) => (
                  <li key={i} className="text-sm leading-[1.55] text-secondary-foreground">
                    {s}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {info.treatments.length > 0 && (
            <div className="flex flex-col gap-2">
              <h4 className="font-mono text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                Treatment
              </h4>
              <ol className="flex flex-col gap-1 pl-5">
                {info.treatments.map((t: string, i: number) => (
                  <li key={i} className="text-sm leading-[1.55] text-secondary-foreground">
                    {t}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </article>
  )
}
