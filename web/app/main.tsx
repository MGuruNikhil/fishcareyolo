import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./routes"
import { ThemeProvider } from "@/components/theme-provider"
import { CameraProvider } from "@/lib/camera/context"
import { DetectionProvider } from "@/lib/detection/context"

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <ThemeProvider>
            <CameraProvider>
                <DetectionProvider>
                    <App />
                </DetectionProvider>
            </CameraProvider>
        </ThemeProvider>
    </StrictMode>,
)
