import { useNavigate } from "react-router-dom"
import { Camera } from "lucide-react"
import { useDetectionContext } from "@/lib/detection/context"
import { Button } from "@/components/ui/button"

export default function NoFishPage() {
  const { currentOutcome } = useDetectionContext()
  const navigate = useNavigate()

  const confidence =
    currentOutcome?.kind === "no_fish"
      ? Math.round((1 - currentOutcome.gateConfidence) * 100)
      : null

  return (
    <div
      className="flex h-screen w-full flex-col items-center justify-center bg-background p-6 transition-colors duration-300"
      role="alert"
      aria-label="No fish detected in the submitted image"
    >
      <div className="flex w-full max-w-[340px] flex-col items-center gap-8 rounded-[2rem] border border-border bg-card p-10 shadow-lg">
        {/* Icon */}
        <div className="relative flex size-24 items-center justify-center" aria-hidden="true">
          {/* Soft ambient glow */}
          <div className="absolute inset-0 rounded-full bg-muted/60 blur-md" />
          {/* Icon ring */}
          <div className="relative flex size-24 items-center justify-center rounded-full border border-border bg-muted/50">
            {/* Fish emoji rendered as text for maximum compatibility */}
            <span className="text-4xl select-none" role="img" aria-label="fish">🐟</span>
          </div>
        </div>

        {/* Text */}
        <div className="flex flex-col items-center gap-3 text-center">
          <h1 className="text-2xl font-semibold tracking-wide text-foreground">
            No fish detected
          </h1>
          <p className="text-sm leading-relaxed text-muted-foreground">
            The image doesn't appear to contain a fish. Please photograph your fish directly
            in well-lit surroundings.
          </p>
          {confidence !== null && (
            <p className="text-xs font-mono text-muted-foreground/60">
              not-fish confidence: {confidence}%
            </p>
          )}
        </div>

        {/* Tips */}
        <ul className="w-full space-y-2 rounded-2xl border border-border/50 bg-muted/30 p-4">
          {[
            "Make sure the fish fills most of the frame",
            "Use good lighting — avoid shadows",
            "Keep the camera steady and close",
          ].map((tip) => (
            <li key={tip} className="flex items-start gap-2 text-xs text-muted-foreground">
              <span className="mt-0.5 shrink-0 text-foreground/40">•</span>
              <span>{tip}</span>
            </li>
          ))}
        </ul>

        {/* CTA */}
        <Button
          id="no-fish-try-again"
          onClick={() => navigate("/")}
          className="flex h-14 w-full items-center justify-center gap-2 rounded-2xl bg-foreground text-background font-semibold tracking-wide shadow-lg transition-all hover:opacity-90 active:scale-95"
        >
          <Camera size={18} aria-hidden="true" />
          <span>Try Again</span>
        </Button>
      </div>
    </div>
  )
}
