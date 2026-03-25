import { useNavigate } from "react-router-dom"
import { RotateCcw, Scan } from "lucide-react"
import { useCameraContext } from "@/lib/camera/context"
import { useEffect } from "react"
import { Button } from "@/components/ui/button"

export default function PreviewPage() {
    const { capturedImageUrl, setCapturedImage } = useCameraContext()
    const navigate = useNavigate()

    useEffect(() => {
        if (!capturedImageUrl) navigate("/", { replace: true })
    }, [capturedImageUrl, navigate])

    const handleRetake = () => {
        setCapturedImage(null)
        navigate("/")
    }

    const handleAnalyse = () => {
        navigate("/analysis")
    }

    if (!capturedImageUrl) return null

    return (
        <div className="relative flex flex-1 flex-col overflow-hidden bg-black">
            {/* Image Preview */}
            <div className="flex flex-1 items-center justify-center">
                <img
                    src={capturedImageUrl}
                    alt="Captured photo ready for analysis"
                    className="block h-full w-full object-contain"
                />
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
