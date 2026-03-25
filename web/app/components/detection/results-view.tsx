import type { InferenceResult } from "@/lib/model/types"
import { AnnotatedImage } from "./annotated-image"
import { ResultsSummary } from "./results-summary"
import { DetectionCard } from "./detection-card"

interface Props {
    imageUrl: string
    result: InferenceResult
    showTimestamp?: boolean
    timestamp?: number
}

function formatTimestamp(ts: number) {
    return new Intl.DateTimeFormat("en-GB", {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(new Date(ts))
}

export function ResultsView({
    imageUrl,
    result,
    showTimestamp,
    timestamp,
}: Props) {
    return (
        <div className="flex flex-1 flex-col md:grid md:grid-cols-[1fr_400px] md:grid-rows-[auto_1fr] md:items-start">
            {showTimestamp && timestamp && (
                <p className="border-b bg-card px-5 py-3 font-mono text-xs uppercase tracking-[0.06em] text-muted-foreground md:col-span-2">
                    Scan — {formatTimestamp(timestamp)}
                </p>
            )}

            {/* Image container - on desktop: fills left column, vertically centered */}
            <div className="md:col-start-1 md:row-start-2 md:self-stretch md:flex md:items-center md:justify-center">
                <AnnotatedImage
                    imageUrl={imageUrl}
                    detections={result.detections}
                />
            </div>

            <div className="flex flex-col gap-4 p-4 pb-8 md:col-start-2 md:row-start-2 md:max-h-[calc(100dvh-120px)] md:overflow-y-auto md:border-l md:px-8 md:py-6 md:pb-10">
                <ResultsSummary detections={result.detections} />

                <h2 className="flex items-center gap-2 font-mono text-sm font-semibold uppercase tracking-[0.08em] text-secondary-foreground">
                    Detections
                    <span className="rounded-full border bg-card px-[7px] py-px text-xs text-muted-foreground">
                        {result.detections.length}
                    </span>
                </h2>

                <div className="flex flex-col gap-2">
                    {result.detections.map((det) => (
                        <DetectionCard key={det.id} detection={det} />
                    ))}
                </div>
            </div>
        </div>
    )
}
