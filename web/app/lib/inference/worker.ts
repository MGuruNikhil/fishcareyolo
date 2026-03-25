import * as ort from "onnxruntime-web"

export interface InferenceRequest {
  id: string
  type: "load" | "run" | "release"
  data?: number[] | string
  dims?: number[]
}

export interface InferenceResponse {
  id: string
  type: "result" | "ready" | "error" | "loading"
  data?: InferenceResult[]
  error?: string
  progress?: number
}

export interface InferenceResult {
  class: string
  confidence: number
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
}

const DISEASE_CLASSES = [
  "bacterial_infection",
  "fungal_infection",
  "healthy",
  "parasite",
  "white_tail",
]

const IMAGE_SIZE = 640
const CONFIDENCE_THRESHOLD = 0.3
const IOU_THRESHOLD = 0.6

declare const ctx: Worker

ort.env.wasm.wasmPaths = "/"

let session: ort.InferenceSession | null = null

function xywh2xyxy(boxes: number[]): number[] {
  const result = []
  for (let i = 0; i < boxes.length; i += 4) {
    const x = boxes[i]
    const y = boxes[i + 1]
    const w = boxes[i + 2]
    const h = boxes[i + 3]
    result.push(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
  }
  return result
}

function computeIoU(box1: number[], box2: number[]): number {
  const x1 = Math.max(box1[0], box2[0])
  const y1 = Math.max(box1[1], box2[1])
  const x2 = Math.min(box1[2], box2[2])
  const y2 = Math.min(box1[3], box2[3])

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
  const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
  const union = area1 + area2 - intersection

  return intersection / (union + 1e-6)
}

function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
  const indices = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.idx)

  const keep: number[] = []
  while (indices.length > 0) {
    const current = indices.shift()!
    keep.push(current)

    const remaining: number[] = []
    for (const idx of indices) {
      const iou = computeIoU(boxes[current], boxes[idx])
      if (iou < iouThreshold) {
        remaining.push(idx)
      }
    }
    indices.length = 0
    indices.push(...remaining)
  }

  return keep
}

function preprocessImage(imageData: ImageData): Float32Array {
  const data = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE)
  const pixels = imageData.data

  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    const r = pixels[i * 4] / 255
    const g = pixels[i * 4 + 1] / 255
    const b = pixels[i * 4 + 2] / 255

    data[i] = r
    data[i + IMAGE_SIZE * IMAGE_SIZE] = g
    data[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = b
  }

  return data
}

function postprocessOutput(output: Float32Array, dims: number[]): InferenceResult[] {
  const numAnchors = dims[2]
  const numClasses = DISEASE_CLASSES.length

  const boxes: number[][] = []
  const scores: number[] = []
  const classIds: number[] = []

  for (let i = 0; i < numAnchors; i++) {
    const baseIdx = i * (numClasses + 5)

    const cx = output[baseIdx]
    const cy = output[baseIdx + 1]
    const w = output[baseIdx + 2]
    const h = output[baseIdx + 3]
    const objConf = output[baseIdx + 4]

    let maxScore = 0
    let classId = 0
    for (let c = 0; c < numClasses; c++) {
      const classConf = output[baseIdx + 5 + c]
      const score = objConf * classConf
      if (score > maxScore) {
        maxScore = score
        classId = c
      }
    }

    if (maxScore > CONFIDENCE_THRESHOLD) {
      boxes.push([cx, cy, w, h])
      scores.push(maxScore)
      classIds.push(classId)
    }
  }

  if (boxes.length === 0) return []

  const xyxyBoxes = xywh2xyxy(boxes.flat())
  const boxArray: number[][] = []
  for (let i = 0; i < xyxyBoxes.length; i += 4) {
    boxArray.push(xyxyBoxes.slice(i, i + 4))
  }

  const keepIndices = nms(boxArray, scores, IOU_THRESHOLD)

  const results: InferenceResult[] = []
  for (const idx of keepIndices) {
    const [x1, y1, x2, y2] = boxArray[idx]
    results.push({
      class: DISEASE_CLASSES[classIds[idx]],
      confidence: scores[idx],
      bbox: {
        x: x1 / IMAGE_SIZE,
        y: y1 / IMAGE_SIZE,
        width: (x2 - x1) / IMAGE_SIZE,
        height: (y2 - y1) / IMAGE_SIZE,
      },
    })
  }

  return results
}

async function loadModel(modelUrl: string): Promise<void> {
  ctx.postMessage({ type: "loading", progress: 0 } as InferenceResponse)

  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    })
    ctx.postMessage({ type: "ready" } as InferenceResponse)
  } catch (e) {
    ctx.postMessage({
      type: "error",
      error: e instanceof Error ? e.message : "Failed to load model",
    } as InferenceResponse)
  }
}

async function runInference(imageData: ImageData): Promise<void> {
  if (!session) {
    ctx.postMessage({
      type: "error",
      error: "Model not loaded",
    } as InferenceResponse)
    return
  }

  try {
    const inputData = preprocessImage(imageData)
    const inputTensor = new ort.Tensor("float32", inputData, [1, 3, IMAGE_SIZE, IMAGE_SIZE])

    const results = await session.run({ input: inputTensor })
    const outputTensor = results.output.data as Float32Array
    const outputDims = results.output.dims as number[]

    const detections = postprocessOutput(outputTensor, outputDims)

    ctx.postMessage({
      type: "result",
      data: detections,
    } as InferenceResponse)
  } catch (e) {
    ctx.postMessage({
      type: "error",
      error: e instanceof Error ? e.message : "Inference failed",
    } as InferenceResponse)
  }
}

ctx.onmessage = async (event: MessageEvent<InferenceRequest>) => {
  const { type, data } = event.data

  if (type === "load") {
    await loadModel(data as unknown as string)
  } else if (type === "run") {
    const imageData = new ImageData(new Uint8ClampedArray(data as number[]), IMAGE_SIZE, IMAGE_SIZE)
    await runInference(imageData)
  } else if (type === "release") {
    session = null
    ctx.postMessage({ type: "ready" } as InferenceResponse)
  }
}
