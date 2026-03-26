import { useState, useEffect } from "react"
import { Monitor, Moon, Sun, Trash2 } from "lucide-react"
import { useTheme } from "@/components/theme-provider"
import { clearHistory, getHistoryItems } from "@/lib/history"
import { Button } from "@/components/ui/button"

type Theme = "light" | "dark" | "system"

const THEME_OPTIONS: {
  value: Theme
  label: string
  icon: typeof Sun
}[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
]

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const [historyCount, setHistoryCount] = useState(0)

  useEffect(() => {
    getHistoryItems()
      .then((items) => setHistoryCount(items.length))
      .catch((err) => console.error("Failed to load history count:", err))
  }, [])

  const handleClearHistory = async () => {
    if (
      window.confirm(
        `Delete all ${historyCount} scan${historyCount !== 1 ? "s" : ""}? This cannot be undone.`,
      )
    ) {
      try {
        await clearHistory()
        setHistoryCount(0)
      } catch (err) {
        console.error("Failed to clear history:", err)
        alert("Failed to clear history. Please try again.")
      }
    }
  }

  return (
    <div className="flex flex-1 flex-col bg-background">
      <header className="flex shrink-0 items-center border-b bg-card p-4 md:px-5">
        <h1 className="font-mono text-lg font-semibold text-foreground">Settings</h1>
      </header>

      <div className="flex w-full max-w-2xl flex-col gap-6 p-6 pb-10 md:p-8">
        {/* Appearance */}
        <section className="flex flex-col gap-2" aria-labelledby="theme-heading">
          <h2
            className="px-1 font-mono text-xs font-semibold uppercase tracking-widest text-muted-foreground"
            id="theme-heading"
          >
            Appearance
          </h2>
          <div className="overflow-hidden rounded-lg border bg-card">
            <div className="flex min-h-16 items-center justify-between gap-4 p-4 md:px-5">
              <div className="flex-1 min-w-0">
                <p className="mb-0.5 text-sm font-medium text-foreground">Theme</p>
                <p className="text-xs leading-relaxed text-secondary-foreground">
                  Choose your preferred colour scheme
                </p>
              </div>
              <div className="flex shrink-0 gap-1" role="radiogroup" aria-label="Theme selection">
                {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
                  <label
                    key={value}
                    className={`flex min-h-9 cursor-pointer items-center gap-1.5 whitespace-nowrap rounded border px-3 py-2 text-xs font-medium transition-colors ${
                      theme === value
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-muted text-secondary-foreground hover:bg-accent hover:text-foreground"
                    }`}
                  >
                    <input
                      type="radio"
                      name="theme"
                      value={value}
                      checked={theme === value}
                      onChange={() => setTheme(value)}
                      className="pointer-events-none absolute h-0 w-0 opacity-0"
                      aria-label={`${label} theme`}
                    />
                    <Icon size={15} aria-hidden="true" />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Data */}
        <section className="flex flex-col gap-2" aria-labelledby="data-heading">
          <h2
            className="px-1 font-mono text-xs font-semibold uppercase tracking-widest text-muted-foreground"
            id="data-heading"
          >
            Data
          </h2>
          <div className="overflow-hidden rounded-lg border bg-card">
            <div className="flex min-h-16 items-center justify-between gap-4 p-4 md:px-5">
              <div className="flex-1 min-w-0">
                <p className="mb-0.5 text-sm font-medium text-foreground">Scan history</p>
                <p className="text-xs leading-relaxed text-secondary-foreground">
                  {historyCount} scan
                  {historyCount !== 1 ? "s" : ""} stored locally on this device
                </p>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={handleClearHistory}
                disabled={historyCount === 0}
                aria-label="Clear all scan history"
                className="shrink-0"
              >
                <Trash2 size={14} aria-hidden="true" />
                Clear
              </Button>
            </div>
          </div>
        </section>

        {/* About */}
        <section className="flex flex-col gap-2" aria-labelledby="about-heading">
          <h2
            className="px-1 font-mono text-xs font-semibold uppercase tracking-widest text-muted-foreground"
            id="about-heading"
          >
            About
          </h2>
          <div className="overflow-hidden rounded-lg border bg-card">
            <div className="flex min-h-16 items-center justify-between gap-4 p-4 md:px-5">
              <div className="flex-1 min-w-0">
                <p className="mb-0.5 text-sm font-medium text-foreground">FishCare YOLO</p>
                <p className="text-xs leading-relaxed text-secondary-foreground">
                  Fish disease detection — on-device ML
                </p>
              </div>
              <span className="shrink-0 font-mono text-xs text-muted-foreground">v1.0.0</span>
            </div>
            <div className="h-px bg-border" aria-hidden="true" />
            <div className="flex min-h-16 items-center gap-4 p-4 md:px-5">
              <div className="flex-1">
                <p className="mb-0.5 text-sm font-medium text-foreground">Privacy</p>
                <p className="text-xs leading-relaxed text-secondary-foreground">
                  All analysis runs on your device. No photos or data are ever uploaded.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
