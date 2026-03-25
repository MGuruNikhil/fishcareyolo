import type { InferenceResponse, InferenceResult } from "./worker"
import InferenceWorker from "./worker?worker"

const MODEL_URL = "https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/best.onnx"

export type InferenceStatus = "idle" | "downloading" | "loading" | "ready" | "error"

export interface InferenceState {
  status: InferenceStatus
  progress: number
  error: string | null
}

export type StatusCallback = (state: InferenceState) => void

class InferenceService {
  private worker: Worker | null = null
  private pendingRequests: Map<
    string,
    { resolve: (data: InferenceResult[]) => void; reject: (err: Error) => void }
  > = new Map()
  private statusCallbacks: Set<StatusCallback> = new Set()
  private state: InferenceState = { status: "idle", progress: 0, error: null }
  private modelUrl: string = MODEL_URL

  private updateState(partial: Partial<InferenceState>) {
    this.state = { ...this.state, ...partial }
    this.statusCallbacks.forEach((cb) => cb(this.state))
  }

  onStatusChange(callback: StatusCallback) {
    this.statusCallbacks.add(callback)
    callback(this.state)
    return () => this.statusCallbacks.delete(callback)
  }

  getStatus(): InferenceState {
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

    this.worker = new InferenceWorker()

    this.worker.onmessage = (event: MessageEvent<InferenceResponse>) => {
      const msg = event.data

      if (msg.type === "loading") {
        this.updateState({ progress: msg.progress || 0 })
      } else if (msg.type === "ready") {
        this.updateState({ status: "ready", progress: 100 })
      } else if (msg.type === "result") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.resolve(msg.data || [])
          this.pendingRequests.delete(msg.id)
        }
      } else if (msg.type === "error") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.reject(new Error(msg.error || "Inference failed"))
          this.pendingRequests.delete(msg.id)
        }
        this.updateState({
          status: "error",
          error: msg.error || "Inference failed",
        })
      }
    }

    this.worker.postMessage({ type: "load", id: "load", data: url })
  }

  async run(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<InferenceResult[]> {
    if (this.state.status !== "ready" || !this.worker) {
      throw new Error("Model not ready. Call serve() first.")
    }

    const canvas = document.createElement("canvas")
    canvas.width = 640
    canvas.height = 640
    const ctx = canvas.getContext("2d")
    if (!ctx) {
      throw new Error("Failed to get canvas context")
    }

    ctx.drawImage(imageElement, 0, 0, 640, 640)
    const imageData = ctx.getImageData(0, 0, 640, 640)

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

export const inferenceService = new InferenceService()
