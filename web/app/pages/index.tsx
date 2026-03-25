import { useRef, useEffect, useState, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import { Camera, Image, AlertCircle, RefreshCw } from "lucide-react"
import { useCameraContext } from "@/lib/camera/context"
import { cn } from "@/lib/utils"

export default function CameraPage() {
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const streamRef = useRef<MediaStream | null>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const [permission, setPermission] = useState<
        "pending" | "granted" | "denied"
    >("pending")
    const [cameraError, setCameraError] = useState<string | null>(null)

    const {
        setCapturedImage,
        cameraFacingMode: facingMode,
        setCameraFacingMode: setFacingMode,
    } = useCameraContext()
    const navigate = useNavigate()

    const requestInProgressRef = useRef(false)
    const pendingRequestRef = useRef<"environment" | "user" | null>(null)
    const mountedRef = useRef(true)

    useEffect(() => {
        mountedRef.current = true
        return () => {
            mountedRef.current = false
            streamRef.current?.getTracks().forEach((t) => t.stop())
            streamRef.current = null
        }
    }, [])

    const startCamera = useCallback(
        async (facing: "environment" | "user") => {
            if (requestInProgressRef.current) {
                pendingRequestRef.current = facing
                return
            }
            requestInProgressRef.current = true
            let currentFacing = facing

            while (currentFacing) {
                try {
                    if (streamRef.current) {
                        streamRef.current.getTracks().forEach((t) => t.stop())
                        if (videoRef.current) {
                            videoRef.current.srcObject = null
                        }
                        streamRef.current = null
                        // Give mobile devices a moment to release the camera hardware
                        await new Promise((resolve) =>
                            setTimeout(resolve, 500),
                        )
                    }

                    if (
                        !navigator.mediaDevices ||
                        !navigator.mediaDevices.getUserMedia
                    ) {
                        if (mountedRef.current) {
                            setPermission("denied")
                            setCameraError(
                                "Camera API is unavailable. This usually happens when the site is not served over a secure connection (HTTPS). Please ensure you are using the HTTPS URL.",
                            )
                        }
                        break
                    }

                    let stream: MediaStream
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({
                            video: {
                                facingMode: currentFacing,
                                width: { ideal: 1920 },
                                height: { ideal: 1080 },
                            },
                        })
                    } catch (err: any) {
                        if (
                            err.name === "NotReadableError" ||
                            err.name === "TrackStartError" ||
                            err.name === "OverconstrainedError"
                        ) {
                            await new Promise((resolve) =>
                                setTimeout(resolve, 500),
                            )
                            stream = await navigator.mediaDevices.getUserMedia({
                                video: { facingMode: currentFacing },
                            })
                        } else {
                            throw err
                        }
                    }

                    if (!mountedRef.current) {
                        stream.getTracks().forEach((t) => t.stop())
                        break
                    }

                    if (pendingRequestRef.current) {
                        stream.getTracks().forEach((t) => t.stop())
                        await new Promise((resolve) => setTimeout(resolve, 500))
                    } else {
                        streamRef.current = stream
                        if (videoRef.current) {
                            videoRef.current.srcObject = stream
                        }
                        setPermission("granted")
                        setCameraError(null)
                    }
                } catch (err: unknown) {
                    if (!mountedRef.current) break
                    if (!pendingRequestRef.current) {
                        const error = err as { name?: string }
                        if (error.name === "NotAllowedError") {
                            setPermission("denied")
                            setCameraError(
                                "Camera permission was denied. Please allow camera access in your browser settings, or use the gallery option below.",
                            )
                        } else if (error.name === "NotFoundError") {
                            setPermission("denied")
                            setCameraError(
                                "No camera found on this device. Please use the gallery option to select a photo.",
                            )
                        } else if (
                            error.name === "NotReadableError" ||
                            error.name === "TrackStartError"
                        ) {
                            // Do not set permission to denied so the switch camera button remains visible
                            setCameraError(
                                "Camera is already in use by another application or tab. Please try switching cameras again.",
                            )
                        } else {
                            setPermission("denied")
                            setCameraError(
                                "Unable to access camera. Please use the gallery option to select a photo.",
                            )
                        }
                        break
                    }
                }

                const nextMode = pendingRequestRef.current
                pendingRequestRef.current = null
                currentFacing = nextMode as typeof currentFacing
            }

            requestInProgressRef.current = false
        },
        [],
    )

    useEffect(() => {
        startCamera(facingMode)
    }, [facingMode, startCamera])

    const handleCapture = async () => {
        const video = videoRef.current
        const canvas = canvasRef.current
        if (!video || !canvas) return

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        const ctx = canvas.getContext("2d")
        if (!ctx) return

        if (facingMode === "user") {
            ctx.translate(canvas.width, 0)
            ctx.scale(-1, 1)
        }

        ctx.drawImage(video, 0, 0)

        // Convert to Blob
        const blob = await new Promise<Blob>((resolve, reject) => {
            canvas.toBlob(
                (b) => (b ? resolve(b) : reject(new Error("Blob creation failed"))),
                "image/jpeg",
                0.92,
            )
        })

        setCapturedImage(blob)
        navigate("/preview")
    }

    const handleGallery = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return
        setCapturedImage(file)
        navigate("/preview")
    }

    const toggleCamera = () => {
        setFacingMode(facingMode === "environment" ? "user" : "environment")
    }

    return (
        <div className="relative flex flex-1 flex-col bg-black">
            {/* Viewfinder */}
            <div className="relative flex-1 overflow-hidden bg-black">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={cn(
                        "h-full w-full object-cover",
                        facingMode === "user" && "scale-x-[-1]",
                        permission === "granted" ? "block" : "hidden",
                    )}
                    aria-label="Live camera feed"
                />

                {permission !== "granted" && (
                    <div
                        className="flex h-full flex-col items-center justify-center gap-4 text-muted-foreground"
                        role="status"
                    >
                        <Camera
                            size={40}
                            className="opacity-40"
                            aria-hidden="true"
                        />
                        <p className="text-sm">Camera unavailable</p>
                    </div>
                )}

                {/* Corner brackets for viewfinder aesthetic */}
                <div className="pointer-events-none absolute left-5 top-5 h-6 w-6 border-l-2 border-t-2 border-white/60" />
                <div className="pointer-events-none absolute right-5 top-5 h-6 w-6 border-r-2 border-t-2 border-white/60" />
                <div className="pointer-events-none absolute bottom-5 left-5 h-6 w-6 border-b-2 border-l-2 border-white/60" />
                <div className="pointer-events-none absolute bottom-5 right-5 h-6 w-6 border-b-2 border-r-2 border-white/60" />
            </div>

            {/* Error Banner */}
            {cameraError && (
                <div
                    className="absolute left-0 right-0 top-0 z-30 flex items-start gap-2 border-b border-white/10 bg-red-950/80 p-4 text-sm text-red-200 backdrop-blur-sm"
                    role="alert"
                >
                    <AlertCircle size={16} aria-hidden="true" />
                    <p className="leading-relaxed">{cameraError}</p>
                </div>
            )}

            {/* Controls */}
            <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center justify-between bg-gradient-to-t from-black/80 via-black/40 to-transparent px-8 py-6 pb-[calc(1.5rem+env(safe-area-inset-bottom))] md:px-12">
                <button
                    className="flex min-h-14 min-w-14 flex-col items-center justify-center gap-1 text-white/70 transition-colors hover:text-white"
                    onClick={() => fileInputRef.current?.click()}
                    aria-label="Choose from gallery"
                >
                    <Image size={20} aria-hidden="true" />
                    <span className="text-[11px] font-medium uppercase tracking-wider">
                        Gallery
                    </span>
                </button>

                <button
                    className="flex size-[72px] items-center justify-center rounded-full border-[3px] border-white/80 transition-all hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30 md:size-20"
                    onClick={handleCapture}
                    disabled={permission !== "granted"}
                    aria-label="Capture photo for analysis"
                >
                    <span
                        className="size-[54px] rounded-full bg-white transition-colors active:bg-white/85 md:size-[62px]"
                        aria-hidden="true"
                    />
                </button>

                {permission === "granted" ? (
                    <div className="flex min-w-14 items-center justify-center">
                        <button
                            className="flex size-12 items-center justify-center rounded-full bg-white/15 text-white transition-all hover:bg-white/25 active:scale-95"
                            onClick={toggleCamera}
                            aria-label="Flip camera"
                        >
                            <RefreshCw size={22} aria-hidden="true" />
                        </button>
                    </div>
                ) : (
                    <div className="min-w-14" aria-hidden="true" />
                )}
            </div>

            {/* Hint */}
            <p className="pointer-events-none absolute bottom-[calc(100px+env(safe-area-inset-bottom))] left-0 right-0 z-15 px-4 py-2 text-center text-xs tracking-wide text-white/60 [text-shadow:0_1px_2px_rgba(0,0,0,0.8)]">
                Position the fish clearly in frame. Ensure good lighting.
            </p>

            <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleGallery}
                aria-label="Upload image from gallery"
            />
        </div>
    )
}
