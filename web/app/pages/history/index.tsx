import { useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { Clock, ChevronRight, Scan } from "lucide-react"
import { getHistoryItems, revokeHistoryItemUrls } from "@/lib/history"
import type { HistoryItem } from "@/lib/history/types"
import { DISEASE_INFO } from "@/lib/model/disease"
import { getSeverityMeta, getWorstSeverity } from "@/lib/model/disease/severity"
import { Button } from "@/components/ui/button"

function formatDate(ts: number) {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(ts))
}

function getSummary(item: HistoryItem) {
  const diseases = item.results.detections.filter((d) => {
    const severity = DISEASE_INFO[d.diseaseClass].severity
    return severity !== "healthy" && severity !== "low"
  })
  if (diseases.length === 0) return "No diseases detected"
  if (diseases.length === 1) return DISEASE_INFO[diseases[0].diseaseClass].displayName
  return `${diseases.length} diseases detected`
}

export default function HistoryIndexPage() {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    let mounted = true

    getHistoryItems()
      .then((items) => {
        if (mounted) {
          setHistory(items)
          setLoading(false)
        }
      })
      .catch((err) => {
        console.error("Failed to load history:", err)
        if (mounted) setLoading(false)
      })

    return () => {
      mounted = false
      // Revoke Object URLs to prevent memory leaks
      history.forEach(revokeHistoryItemUrls)
    }
  }, [])

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col bg-background">
      <header className="flex shrink-0 items-center justify-between border-b bg-card p-4 md:px-5">
        <h1 className="font-mono text-lg font-semibold text-foreground">History</h1>
        <span className="rounded-full border bg-card px-2 py-0.5 font-mono text-xs text-muted-foreground">
          {history.length} scans
        </span>
      </header>

      {history.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 p-10 text-center">
          <Clock size={36} className="text-muted-foreground opacity-50" />
          <p className="text-lg font-semibold text-foreground">No scans yet</p>
          <p className="max-w-[280px] text-sm leading-relaxed text-secondary-foreground">
            Your scan history will appear here after you analyse a fish photo.
          </p>
          <Button onClick={() => navigate("/")} className="mt-2 min-h-11">
            <Scan size={16} />
            Start scanning
          </Button>
        </div>
      ) : (
        <ul className="flex flex-col md:gap-2 md:p-4">
          {history.map((item) => {
            const severity = getWorstSeverity(item.results.detections)
            const meta = getSeverityMeta(severity)
            return (
              <li key={item.id}>
                <button
                  className="flex min-h-[72px] w-full items-center gap-4 border-b bg-card p-3 text-left transition-colors hover:bg-accent/50 md:rounded-lg md:border md:p-4"
                  onClick={() => navigate(`/history/${item.id}`)}
                  aria-label={`View scan from ${formatDate(item.timestamp)}: ${getSummary(item)}`}
                >
                  <div className="h-14 w-14 shrink-0 overflow-hidden rounded-md border bg-muted md:h-16 md:w-16">
                    <img
                      src={item.processedImageUrl}
                      alt=""
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 shrink-0 rounded-full"
                        style={{
                          background: meta.color,
                        }}
                      />
                      <p className="overflow-hidden text-ellipsis whitespace-nowrap text-sm font-medium text-foreground">
                        {getSummary(item)}
                      </p>
                    </div>
                    <time
                      className="font-mono text-xs text-muted-foreground"
                      dateTime={new Date(item.timestamp).toISOString()}
                    >
                      {formatDate(item.timestamp)}
                    </time>
                    <span
                      className="w-fit rounded-full border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider"
                      style={{
                        background: meta.bg,
                        color: meta.color,
                        borderColor: meta.border,
                      }}
                    >
                      {meta.label}
                    </span>
                  </div>
                  <ChevronRight size={16} className="shrink-0 text-muted-foreground" />
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
