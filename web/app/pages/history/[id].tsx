import { useParams, useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { ArrowLeft } from "lucide-react"
import { getHistoryItem, revokeHistoryItemUrls } from "@/lib/history"
import type { HistoryItem } from "@/lib/history/types"
import { ResultsView } from "@/components/detection/results-view"
import { Button } from "@/components/ui/button"

export default function HistoryDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [item, setItem] = useState<HistoryItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)

  useEffect(() => {
    if (!id) {
      setNotFound(true)
      setLoading(false)
      return
    }

    let mounted = true

    getHistoryItem(id)
      .then((result) => {
        if (!mounted) return
        if (result) {
          setItem(result)
        } else {
          setNotFound(true)
        }
        setLoading(false)
      })
      .catch((err) => {
        console.error("Failed to load history item:", err)
        if (mounted) {
          setNotFound(true)
          setLoading(false)
        }
      })

    return () => {
      mounted = false
      if (item) {
        revokeHistoryItemUrls(item)
      }
    }
  }, [id])

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    )
  }

  if (notFound || !item) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Scan not found.</p>
        <Button onClick={() => navigate("/history")}>
          <ArrowLeft size={16} />
          Back to History
        </Button>
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col bg-background">
      <header className="flex shrink-0 items-center justify-between border-b bg-card p-4 md:px-5">
        <button
          className="flex items-center gap-2 text-sm text-secondary-foreground transition-colors hover:text-foreground"
          onClick={() => navigate("/history")}
          aria-label="Back to history"
        >
          <ArrowLeft size={18} />
          <span>History</span>
        </button>
        <h1 className="font-mono text-lg font-semibold text-foreground">Scan Detail</h1>
      </header>
      <ResultsView
        imageUrl={item.processedImageUrl}
        result={item.results}
        showTimestamp
        timestamp={item.timestamp}
      />
    </div>
  )
}
