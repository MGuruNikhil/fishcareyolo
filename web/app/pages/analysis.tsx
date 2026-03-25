import { useEffect, useRef } from "react"
import { useNavigate } from "react-router-dom"
import { useCameraContext } from "@/lib/camera/context"
import { useDetectionContext } from "@/lib/detection/context"

const STEPS = [
    "Loading image data",
    "Running disease detection model",
    "Scoring detections",
    "Preparing results",
]

// Dummy analysis function - will be replaced with actual ML inference
async function runDummyAnalysis(): Promise<any> {
    // Simulate inference time
    await new Promise((resolve) => setTimeout(resolve, 1800))

    // Return dummy results for now
    return {
        detections: [],
        inferenceTimeMs: 1800,
    }
}

export default function AnalysisPage() {
    const { capturedImage } = useCameraContext()
    const { setCurrentResult } = useDetectionContext()
    const navigate = useNavigate()
    const ran = useRef(false)

    useEffect(() => {
        if (!capturedImage) {
            navigate("/", { replace: true })
            return
        }
        if (ran.current) return
        ran.current = true

        runDummyAnalysis().then((result) => {
            setCurrentResult(result)
            // TODO: Save to history here
            navigate("/results", { replace: true })
        })
    }, [capturedImage, navigate, setCurrentResult])

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

                <p className="font-mono text-xl font-semibold tracking-wide">
                    Analysing
                </p>

                {/* Steps */}
                <ul className="flex w-full flex-col gap-3" aria-hidden="true">
                    {STEPS.map((step, i) => (
                        <li
                            key={i}
                            className="flex items-center gap-3 font-mono text-sm text-muted-foreground opacity-0"
                            style={{
                                animation: `fadeIn 0.4s ease forwards ${i * 0.4}s`,
                            }}
                        >
                            <span className="size-1.5 shrink-0 rounded-full bg-primary opacity-0" style={{
                                animation: `fadeIn 0.4s ease forwards ${i * 0.4}s`,
                            }} />
                            <span>{step}</span>
                        </li>
                    ))}
                </ul>

                <p className="font-mono text-xs text-muted-foreground tracking-wide">
                    Running on-device — no data is transmitted
                </p>
            </div>

            <style>{`
                @keyframes fadeIn {
                    to {
                        opacity: 1;
                        color: hsl(var(--secondary-foreground));
                    }
                }
            `}</style>
        </div>
    )
}
