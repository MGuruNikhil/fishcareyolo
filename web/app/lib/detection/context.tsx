import React, { createContext, useContext, useState } from "react"
import type { AnalysisOutcome } from "@/lib/model/types"

interface DetectionState {
  currentOutcome: AnalysisOutcome | null
  setCurrentOutcome: (outcome: AnalysisOutcome | null) => void
}

const DetectionContext = createContext<DetectionState | null>(null)

export function DetectionProvider({ children }: { children: React.ReactNode }) {
  const [currentOutcome, setCurrentOutcome] = useState<AnalysisOutcome | null>(null)

  return (
    <DetectionContext.Provider
      value={{
        currentOutcome,
        setCurrentOutcome,
      }}
    >
      {children}
    </DetectionContext.Provider>
  )
}

export function useDetectionContext() {
  const ctx = useContext(DetectionContext)
  if (!ctx) throw new Error("useDetectionContext must be inside DetectionProvider")
  return ctx
}
