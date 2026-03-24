/**
 * Type definitions for fish disease detection model.
 * Adapted from mina-fork for web use.
 */

/** Disease classification types */
export type DiseaseClass =
    | "bacterial_infection"
    | "fungal_infection"
    | "healthy"
    | "parasite"
    | "white_tail"

/** List of all disease classes (matches model output order) */
export const DISEASE_CLASSES: DiseaseClass[] = [
    "bacterial_infection",
    "fungal_infection",
    "healthy",
    "parasite",
    "white_tail",
]

/** Bounding box coordinates (normalized 0-1 relative to image dimensions) */
export interface BoundingBox {
    x: number
    y: number
    width: number
    height: number
}

/** Single disease detection result */
export interface Detection {
    id: string
    diseaseClass: DiseaseClass
    confidence: number
    boundingBox: BoundingBox
}

/** Inference result from model */
export interface InferenceResult {
    detections: Detection[]
    inferenceTimeMs: number
}

/** Validation helper: check if value is a valid disease class */
export function isValidDiseaseClass(value: string): value is DiseaseClass {
    return DISEASE_CLASSES.includes(value as DiseaseClass)
}

/** Validation helper: check if confidence is in valid range */
export function isValidConfidence(value: number): boolean {
    return value >= 0 && value <= 1
}

/** Validation helper: check if bounding box is valid */
export function isValidBoundingBox(bbox: BoundingBox): boolean {
    return (
        bbox.x >= 0 &&
        bbox.x <= 1 &&
        bbox.y >= 0 &&
        bbox.y <= 1 &&
        bbox.width >= 0 &&
        bbox.width <= 1 &&
        bbox.height >= 0 &&
        bbox.height <= 1 &&
        bbox.x + bbox.width <= 1 + 1e-6 &&
        bbox.y + bbox.height <= 1 + 1e-6
    )
}
