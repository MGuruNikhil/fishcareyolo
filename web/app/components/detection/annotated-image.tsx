import { useRef, useEffect, useState } from "react"
import type { Detection } from "@/lib/model/types"
import { getBoundingBoxColor, getDiseaseInfo } from "@/lib/model/disease"

interface Props {
    imageUrl: string
    detections: Detection[]
}

export function AnnotatedImage({ imageUrl, detections }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const imgRef = useRef<HTMLImageElement>(null)
    const [imgSize, setImgSize] = useState({ w: 0, h: 0 })

    useEffect(() => {
        const img = new Image()
        img.onload = () => {
            setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
        }
        img.src = imageUrl
    }, [imageUrl])

    // Draw bounding boxes on canvas overlay
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas || imgSize.w === 0) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        ctx.clearRect(0, 0, canvas.width, canvas.height)

        detections.forEach((det) => {
            const info = getDiseaseInfo(det.diseaseClass)
            const color = getBoundingBoxColor(det.diseaseClass)
            const x = det.boundingBox.x * canvas.width
            const y = det.boundingBox.y * canvas.height
            const w = det.boundingBox.width * canvas.width
            const h = det.boundingBox.height * canvas.height

            ctx.strokeStyle = color
            ctx.lineWidth = 2.5
            ctx.strokeRect(x, y, w, h)

            // Label background with display name
            const confidence = (det.confidence * 100).toFixed(0)
            const labelText = `${info.displayName} ${confidence}%`
            ctx.font = "bold 12px monospace"
            const textW = ctx.measureText(labelText).width
            const labelH = 20
            const labelY = y > labelH + 4 ? y - labelH - 2 : y + h + 2

            ctx.fillStyle = color
            ctx.fillRect(x, labelY, textW + 10, labelH)

            ctx.fillStyle = "#000"
            ctx.fillText(labelText, x + 5, labelY + 14)
        })
    }, [detections, imgSize])

    const aspectRatio = imgSize.w > 0 ? imgSize.w / imgSize.h : 16 / 9

    return (
        <div
            className="relative w-full overflow-hidden bg-black max-h-[50vh] md:max-h-full"
            style={{ aspectRatio: `${aspectRatio}` }}
            role="img"
            aria-label={`Fish scan with ${detections.length} detection${detections.length !== 1 ? "s" : ""} annotated`}
        >
            <img
                ref={imgRef}
                src={imageUrl}
                alt=""
                className="block h-full w-full object-contain"
                onLoad={(e) => {
                    const img = e.currentTarget
                    setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
                }}
            />
            <canvas
                ref={canvasRef}
                width={imgSize.w || 1}
                height={imgSize.h || 1}
                className="pointer-events-none absolute inset-0 h-full w-full"
                aria-hidden="true"
            />
        </div>
    )
}
