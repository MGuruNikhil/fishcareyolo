import { useNavigate } from "react-router-dom"
import { RotateCcw, Scan } from "lucide-react"
import { useCameraContext } from "@/lib/camera/context"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"

export default function PreviewPage() {
  const { capturedImageUrl, setCapturedImage } = useCameraContext()
  const navigate = useNavigate()
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    if (!capturedImageUrl) navigate("/", { replace: true })
  }, [capturedImageUrl, navigate])

  useEffect(() => {
    if (!capturedImageUrl) return

    const img = new Image()
    img.onload = () => {
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
    }
    img.src = capturedImageUrl
  }, [capturedImageUrl])

  const handleRetake = () => {
    setCapturedImage(null)
    navigate("/")
  }

  const handleAnalyse = () => {
    navigate("/analysis")
  }

  if (!capturedImageUrl) return null

  const aspectRatio = imgSize.w > 0 ? imgSize.w / imgSize.h : 16 / 9

  return (
    <div className="relative flex h-[calc(100dvh-60px)] flex-col overflow-hidden bg-black md:h-auto md:flex-1">
      {/* Image Preview */}
      <div className="flex min-h-0 flex-1 items-center justify-center px-4 pb-[calc(7.5rem+env(safe-area-inset-bottom))] pt-4 md:px-8 md:pb-[calc(8.5rem+env(safe-area-inset-bottom))] md:pt-8">
        <div className="relative w-full overflow-hidden bg-black max-h-[50vh] md:max-h-full" style={{ aspectRatio }}>
          <img
            src={capturedImageUrl}
            alt="Captured photo ready for analysis"
            className="block h-full w-full object-contain"
            onLoad={(e) => {
              const img = e.currentTarget
              setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
            }}
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="absolute bottom-0 left-0 right-0 z-20 flex gap-3 bg-gradient-to-t from-black/80 via-black/40 to-transparent px-6 py-5 pb-[calc(1.5rem+env(safe-area-inset-bottom))] md:px-12 md:py-6">
        <Button
          variant="outline"
          size="lg"
          onClick={handleRetake}
          className="flex-1 gap-2 border-white/20 bg-white/10 text-white backdrop-blur-sm hover:bg-white/20 hover:text-white"
        >
          <RotateCcw size={18} aria-hidden="true" />
          <span>Retake</span>
        </Button>
        <Button
          size="lg"
          onClick={handleAnalyse}
          className="flex-[2] gap-2 bg-primary text-primary-foreground shadow-lg hover:bg-primary/90"
        >
          <Scan size={18} aria-hidden="true" />
          <span>Analyse</span>
        </Button>
      </div>
    </div>
  )
}
