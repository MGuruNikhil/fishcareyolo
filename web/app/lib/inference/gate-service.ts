import type { GateResponse } from "./gate-worker"
import GateWorker from "./gate-worker?worker"

export interface GateResult {
  isFish: boolean
  confidence: number
}

export type GateStatus = "idle" | "loading" | "ready" | "error"

export interface GateState {
  status: GateStatus
  progress: number
  error: string | null
}

export type GateStatusCallback = (state: GateState) => void

function getDefaultGateModelUrl(): string {
  const configured = import.meta.env.VITE_GATE_MODEL_URL

  if (typeof configured === "string" && configured.trim().length > 0) {
    return configured.trim()
  }

  return `${import.meta.env.BASE_URL}model/fish_gate.onnx`
}

const GATE_MODEL_URL = getDefaultGateModelUrl()
const GATE_SIZE = 224

class GateService {
  private worker: Worker | null = null
  private pendingRequests: Map<
    string,
    {
      resolve: (data: GateResult) => void
      reject: (err: Error) => void
    }
  > = new Map()
  private statusCallbacks: Set<GateStatusCallback> = new Set()
  private state: GateState = { status: "idle", progress: 0, error: null }
  private modelUrl: string = GATE_MODEL_URL

  private updateState(partial: Partial<GateState>) {
    this.state = { ...this.state, ...partial }
    this.statusCallbacks.forEach((cb) => cb(this.state))
  }

  onStatusChange(callback: GateStatusCallback) {
    this.statusCallbacks.add(callback)
    callback(this.state)
    return () => this.statusCallbacks.delete(callback)
  }

  getStatus(): GateState {
    return this.state
  }

  setModelUrl(url: string) {
    this.modelUrl = url
  }

  async serve(): Promise<void> {
    if (this.state.status === "ready") {
      return
    }

    this.updateState({ status: "loading", progress: 0, error: null })
    this.initWorker(this.modelUrl)
  }

  private initWorker(url: string) {
    if (this.worker) {
      this.worker.terminate()
    }

    this.worker = new GateWorker()

    this.worker.onerror = (event: ErrorEvent) => {
      const message =
        event.message && event.message.length > 0
          ? event.message
          : "Gate worker crashed while loading the model"

      this.updateState({ status: "error", error: message })
    }

    this.worker.onmessageerror = () => {
      this.updateState({
        status: "error",
        error: "Gate worker failed to process a message",
      })
    }

    this.worker.onmessage = (event: MessageEvent<GateResponse>) => {
      const msg = event.data

      if (msg.type === "loading") {
        this.updateState({ progress: msg.progress ?? 0 })
      } else if (msg.type === "ready") {
        this.updateState({ status: "ready", progress: 100 })
      } else if (msg.type === "result") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.resolve(msg.data!)
          this.pendingRequests.delete(msg.id)
        }
      } else if (msg.type === "error") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.reject(new Error(msg.error || "Gate inference failed"))
          this.pendingRequests.delete(msg.id)
        }
        this.updateState({
          status: "error",
          error: msg.error || "Gate inference failed",
        })
      }
    }

    this.worker.postMessage({ type: "load", id: "load", data: url })
  }

  async run(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<GateResult> {
    if (this.state.status !== "ready" || !this.worker) {
      throw new Error("Gate model not ready. Call serve() first.")
    }

    // Scale-to-fill resize to 224×224 (no letterbox needed for classifier)
    const canvas = document.createElement("canvas")
    canvas.width = GATE_SIZE
    canvas.height = GATE_SIZE
    const canvasCtx = canvas.getContext("2d")
    if (!canvasCtx) {
      throw new Error("Failed to get canvas context for gate preprocessing")
    }

    canvasCtx.drawImage(imageElement, 0, 0, GATE_SIZE, GATE_SIZE)
    const imageData = canvasCtx.getImageData(0, 0, GATE_SIZE, GATE_SIZE)

    const id = crypto.randomUUID()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })

      this.worker!.postMessage({
        type: "run",
        id,
        data: Array.from(imageData.data),
      })
    })
  }

  unserved(): void {
    if (this.worker) {
      this.worker.postMessage({ type: "release", id: "release", data: [] })
      this.worker.terminate()
      this.worker = null
    }
    this.updateState({ status: "idle", progress: 0, error: null })
  }
}

export const gateService = new GateService()
