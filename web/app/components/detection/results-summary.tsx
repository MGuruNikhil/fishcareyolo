import { AlertCircle, CheckCircle } from "lucide-react"
import type { Detection } from "@/lib/model/types"
import { getDiseaseInfo } from "@/lib/model/disease"

interface Props {
    detections: Detection[]
}

export function ResultsSummary({ detections }: Props) {
    // Check if all detections are healthy (severity === "healthy")
    const allHealthy = detections.every((d) => {
        const info = getDiseaseInfo(d.diseaseClass)
        return info.severity === "healthy"
    })
    
    const highOrMediumCount = detections.filter((d) => {
        const info = getDiseaseInfo(d.diseaseClass)
        return info.severity === "high" || info.severity === "medium"
    }).length

    if (allHealthy) {
        return (
            <div
                className="flex items-center gap-2 rounded-md border px-4 py-3 text-sm font-medium"
                style={{
                    backgroundColor: "var(--healthy-bg)",
                    borderColor: "var(--healthy-border)",
                    color: "var(--healthy)",
                }}
                role="status"
            >
                <CheckCircle size={18} aria-hidden="true" />
                <span>No diseases detected — fish appears healthy</span>
            </div>
        )
    }

    const borderColor =
        highOrMediumCount > 0 ? "var(--medium-border)" : "var(--low-border)"
    const bgColor =
        highOrMediumCount > 0 ? "var(--medium-bg)" : "var(--low-bg)"
    const iconColor =
        highOrMediumCount > 0 ? "var(--medium)" : "var(--low)"

    return (
        <div
            className="flex items-start gap-2 rounded-md border px-4 py-3 text-sm font-medium text-foreground"
            style={{
                borderColor,
                backgroundColor: bgColor,
            }}
            role="status"
        >
            <AlertCircle
                size={18}
                className="shrink-0"
                style={{ color: iconColor }}
                aria-hidden="true"
            />
            <span>
                {detections.length} detection
                {detections.length !== 1 ? "s" : ""} found
                {highOrMediumCount > 0
                    ? ` — ${highOrMediumCount} require urgent attention`
                    : ""}
            </span>
        </div>
    )
}
