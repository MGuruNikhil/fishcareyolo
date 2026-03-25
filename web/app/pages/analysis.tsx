import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useCameraContext } from "@/lib/camera/context"
import { useDetectionContext } from "@/lib/detection/context"
import { inferenceService, transformResults } from "@/lib/inference"
import type { InferenceStatus } from "@/lib/inference"
import { saveHistoryItem } from "@/lib/history"
import { loadImageFromBlob, createAnnotatedImage } from "@/lib/utils/image"

type AnalysisStep =
  | "loading-image"
  | "loading-model"
  | "running-inference"
  | "processing-results"
  | "saving"

const STEP_LABELS: Record<AnalysisStep, string> = {
  "loading-image": "Loading image data",
  "loading-model": "Loading disease detection model",
  "running-inference": "Running disease detection",
  "processing-results": "Processing detections",
  saving: "Saving to history",
}

const STEPS: AnalysisStep[] = [
  "loading-image",
  "loading-model",
  "running-inference",
  "processing-results",
  "saving",
]

export default function AnalysisPage() {
  const { capturedImage } = useCameraContext()
  const { setCurrentResult } = useDetectionContext()
  const navigate = useNavigate()
  const ran = useRef(false)
  const [currentStep, setCurrentStep] = useState<AnalysisStep>("loading-image")
  const [modelStatus, setModelStatus] = useState<InferenceStatus>("idle")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!capturedImage) {
      navigate("/", { replace: true })
      return
    }
    if (ran.current) return
    ran.current = true

    runAnalysis(capturedImage)
  }, [capturedImage, navigate, setCurrentResult])

  async function runAnalysis(imageBlob: Blob) {
    try {
      // Step 1: Load image
      setCurrentStep("loading-image")
      const img = await loadImageFromBlob(imageBlob)

      // Step 2: Ensure model is loaded
      setCurrentStep("loading-model")

      // Subscribe to model loading status
      const unsubscribe = inferenceService.onStatusChange((state) => {
        setModelStatus(state.status)
        if (state.status === "error") {
          throw new Error(state.error || "Failed to load model")
        }
      })

      try {
        await inferenceService.serve()

        // Wait for model to be ready
        const status = inferenceService.getStatus()
        if (status.status !== "ready") {
          // Wait for ready status
          await new Promise<void>((resolve, reject) => {
            const checkStatus = inferenceService.onStatusChange((state) => {
              if (state.status === "ready") {
                checkStatus()
                resolve()
              } else if (state.status === "error") {
                checkStatus()
                reject(new Error(state.error || "Failed to load model"))
              }
            })
          })
        }
      } finally {
        unsubscribe()
      }

      // Step 3: Run inference
      setCurrentStep("running-inference")
      const startTime = performance.now()
      const rawResults = await inferenceService.run(img)
      const inferenceTimeMs = performance.now() - startTime

      // Step 4: Transform results
      setCurrentStep("processing-results")
      const result = transformResults(rawResults, inferenceTimeMs)

      // Step 5: Create annotated image and save to history
      setCurrentStep("saving")
      const annotatedImage = await createAnnotatedImage(imageBlob, result.detections)

      await saveHistoryItem({
        timestamp: Date.now(),
        originalImage: imageBlob,
        processedImage: annotatedImage,
        results: result,
      })

      // Set result in context and navigate to results page
      setCurrentResult(result)
      navigate("/results", { replace: true })
    } catch (err) {
      console.error("Analysis failed:", err)
      setError(err instanceof Error ? err.message : "Analysis failed")

      // Navigate back to preview after a short delay so user can see error
      setTimeout(() => {
        navigate("/preview", { replace: true })
      }, 2000)
    }
  }

  const currentStepIndex = STEPS.indexOf(currentStep)

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center p-8" role="alert">
        <div className="flex w-full max-w-xs flex-col items-center gap-6">
          <div className="relative size-[72px] flex items-center justify-center" aria-hidden="true">
            <div className="text-destructive text-4xl">!</div>
          </div>

          <p className="font-mono text-xl font-semibold tracking-wide text-destructive">
            Analysis Failed
          </p>

          <p className="text-center text-sm text-muted-foreground">{error}</p>

          <p className="font-mono text-xs text-muted-foreground tracking-wide">
            Returning to preview...
          </p>
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex flex-1 items-center justify-center p-8"
      role="status"
      aria-label="Analysing image, please wait"
    >
      <div className="flex w-full max-w-xs flex-col items-center gap-6">
        {/* Spinner */}
        <div className="relative size-[72px]" aria-hidden="true">
          <div className="absolute inset-0 rounded-full border-[3px] border-transparent border-t-primary animate-spin" />
        </div>

        <p className="font-mono text-xl font-semibold tracking-wide">Analysing</p>

        {/* Steps */}
        <ul className="flex w-full flex-col gap-3" aria-hidden="true">
          {STEPS.map((step, i) => {
            const isActive = i === currentStepIndex
            const isCompleted = i < currentStepIndex

            return (
              <li
                key={step}
                className={`flex items-center gap-3 font-mono text-sm transition-all duration-300 ${
                  isCompleted
                    ? "text-secondary-foreground"
                    : isActive
                      ? "text-foreground"
                      : "text-muted-foreground opacity-50"
                }`}
              >
                <span
                  className={`size-1.5 shrink-0 rounded-full transition-all duration-300 ${
                    isCompleted || isActive ? "bg-primary" : "bg-muted-foreground"
                  }`}
                />
                <span>{STEP_LABELS[step]}</span>
                {isActive && step === "loading-model" && modelStatus === "loading" && (
                  <span className="text-xs text-muted-foreground">(downloading...)</span>
                )}
              </li>
            )
          })}
        </ul>

        <p className="font-mono text-xs text-muted-foreground tracking-wide">
          Running on-device — no data is transmitted
        </p>
      </div>
    </div>
  )
}
