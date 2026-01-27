/**
 * React context for model state management.
 *
 * Provides model state to the entire app and auto-initializes on mount.
 */

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useState,
    type ReactNode,
} from "react"
import {
    forceUpdateModel,
    getLocalModelPath,
    initializeModel,
} from "@/lib/model/manager"
import {
    runInference as runModelInference,
    type InferenceResult,
} from "@/lib/model/inference"
import type { ModelState } from "@/lib/model/types"

interface ModelContextValue extends ModelState {
    /** Re-download the model from scratch */
    forceUpdate: () => Promise<void>
    /** Get the local file path to the model (for TFLite) */
    getModelPath: () => Promise<string | null>
    /** Run inference on an image */
    runInference: (imageUri: string) => Promise<InferenceResult | null>
}

const initialState: ModelState = {
    isReady: false,
    isLoading: true,
    progress: 0,
    error: null,
    metadata: null,
}

const ModelContext = createContext<ModelContextValue | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<ModelState>(initialState)

    // Initialize model on mount
    useEffect(() => {
        let cancelled = false

        async function init() {
            setState((prev) => ({ ...prev, isLoading: true, progress: 0 }))

            const result = await initializeModel((progress) => {
                if (!cancelled) {
                    setState((prev) => ({ ...prev, progress }))
                }
            })

            if (!cancelled) {
                setState(result)
            }
        }

        init()

        return () => {
            cancelled = true
        }
    }, [])

    const forceUpdate = useCallback(async () => {
        setState((prev) => ({
            ...prev,
            isLoading: true,
            progress: 0,
            error: null,
        }))

        const result = await forceUpdateModel((progress) => {
            setState((prev) => ({ ...prev, progress }))
        })

        setState(result)
    }, [])

    const getModelPath = useCallback(async () => {
        return getLocalModelPath()
    }, [])

    const runInference = useCallback(
        async (imageUri: string): Promise<InferenceResult | null> => {
            try {
                // Check if model is ready
                if (!state.isReady || state.error) {
                    console.warn("Model not ready for inference:", state.error)
                    return null
                }

                // Get model path
                const modelPath = await getLocalModelPath()
                if (!modelPath) {
                    console.warn("Model file not found")
                    return null
                }

                // Run inference
                const result = await runModelInference(imageUri, modelPath)
                return result
            } catch (error) {
                const message =
                    error instanceof Error ? error.message : String(error)
                console.error("Inference error:", message)
                return null
            }
        },
        [state.isReady, state.error],
    )

    return (
        <ModelContext.Provider
            value={{ ...state, forceUpdate, getModelPath, runInference }}
        >
            {children}
        </ModelContext.Provider>
    )
}

/** Hook to access model state and actions */
export function useModel(): ModelContextValue {
    const context = useContext(ModelContext)
    if (!context) {
        throw new Error("useModel must be used within a ModelProvider")
    }
    return context
}
