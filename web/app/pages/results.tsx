import { useNavigate } from "react-router-dom"
import { Camera } from "lucide-react"
import { useDetectionContext } from "@/lib/detection/context"
import { useCameraContext } from "@/lib/camera/context"
import { ResultsView } from "@/components/detection/results-view"
import { Button } from "@/components/ui/button"

export default function ResultsPage() {
    const { currentResult } = useDetectionContext()
    const { capturedImageUrl } = useCameraContext()
    const navigate = useNavigate()

    if (!currentResult || !capturedImageUrl) {
        return (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 text-muted-foreground">
                <p>No results available.</p>
                <Button 
                    onClick={() => navigate("/")}
                    className="bg-accent text-white hover:bg-accent/90"
                >
                    <Camera className="h-4 w-4" />
                    Go to Camera
                </Button>
            </div>
        )
    }

    return (
        <div className="flex flex-1 flex-col bg-background">
            <header className="flex shrink-0 items-center justify-between border-b bg-card px-4 py-4 md:px-5">
                <h1 className="font-mono text-lg font-semibold text-foreground">
                    Results
                </h1>
                <div className="flex items-center gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        className="min-h-9"
                        onClick={() => navigate("/")}
                    >
                        <Camera className="h-4 w-4" />
                        New scan
                    </Button>
                </div>
            </header>
            <ResultsView
                imageUrl={capturedImageUrl}
                result={currentResult}
            />
        </div>
    )
}
