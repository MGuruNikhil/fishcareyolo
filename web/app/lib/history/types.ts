/**
 * Type definitions for history storage.
 * Adapted from mina-fork for web IndexedDB storage.
 */

import type { InferenceResult } from "@/lib/model/types"

/**
 * History item as exposed to components.
 * Images are represented as Object URLs created from stored Blobs.
 */
export interface HistoryItem {
    id: string
    timestamp: number
    originalImageUrl: string
    processedImageUrl: string
    results: InferenceResult
}

/**
 * History item as stored in IndexedDB.
 * Images are stored as ArrayBuffers (more efficient and universally serializable).
 */
export interface StoredHistoryItem {
    id: string
    timestamp: number
    originalImage: ArrayBuffer
    processedImage: ArrayBuffer
    originalImageType: string
    processedImageType: string
    results: InferenceResult
}

/**
 * Input for creating a new history item.
 * ID is generated automatically by the storage layer.
 */
export interface HistoryItemInput {
    timestamp: number
    originalImage: Blob
    processedImage: Blob
    results: InferenceResult
}
