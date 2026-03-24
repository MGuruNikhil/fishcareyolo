/**
 * IndexedDB storage for detection history.
 * Adapted from mina-fork's MMKV + FileSystem storage to use browser IndexedDB.
 *
 * Key differences from React Native version:
 * - Uses IndexedDB instead of MMKV for metadata
 * - Stores images as ArrayBuffers (universally serializable)
 * - Returns Object URLs for use in <img> tags
 */

import type { HistoryItem, StoredHistoryItem, HistoryItemInput } from "./types"
import { generateUUID } from "@/lib/utils/uuid"

const DB_NAME = "fishcare_history"
const DB_VERSION = 1
const STORE_NAME = "history_items"

let dbInstance: IDBDatabase | null = null

/**
 * Initialize the IndexedDB database.
 * Must be called before any other storage operations.
 *
 * @returns Promise that resolves when database is ready
 */
export async function initHistoryDB(): Promise<void> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION)

        request.onerror = () => reject(request.error)

        request.onsuccess = () => {
            dbInstance = request.result
            resolve()
        }

        request.onupgradeneeded = (event) => {
            const db = (event.target as IDBOpenDBRequest).result
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                const store = db.createObjectStore(STORE_NAME, { keyPath: "id" })
                // Create index on timestamp for efficient sorted queries
                store.createIndex("timestamp", "timestamp", { unique: false })
            }
        }
    })
}

/**
 * Get the database instance.
 * Throws if database hasn't been initialized.
 */
function getDB(): IDBDatabase {
    if (!dbInstance) {
        throw new Error("Database not initialized. Call initHistoryDB() first.")
    }
    return dbInstance
}

/**
 * Save a new history item to storage.
 * Converts Blobs to ArrayBuffers for efficient, cross-environment serialization.
 *
 * @param input - History item data (ID will be auto-generated)
 * @returns Promise resolving to the saved item with Object URLs
 */
export async function saveHistoryItem(
    input: HistoryItemInput,
): Promise<HistoryItem> {
    const db = getDB()
    const id = generateUUID()

    // Convert Blobs to ArrayBuffers for IndexedDB storage
    const originalBuffer = await input.originalImage.arrayBuffer()
    const processedBuffer = await input.processedImage.arrayBuffer()

    const stored: StoredHistoryItem = {
        id,
        timestamp: input.timestamp,
        originalImage: originalBuffer,
        processedImage: processedBuffer,
        originalImageType: input.originalImage.type,
        processedImageType: input.processedImage.type,
        results: input.results,
    }

    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite")
        const store = tx.objectStore(STORE_NAME)
        const request = store.add(stored)

        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
            resolve(storedToHistoryItem(stored))
        }
    })
}

/**
 * Get all history items, sorted by timestamp (newest first).
 *
 * @returns Promise resolving to array of history items with Object URLs
 */
export async function getHistoryItems(): Promise<HistoryItem[]> {
    const db = getDB()

    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly")
        const store = tx.objectStore(STORE_NAME)
        const index = store.index("timestamp")
        const request = index.openCursor(null, "prev") // descending order

        const items: HistoryItem[] = []

        request.onerror = () => reject(request.error)
        request.onsuccess = (event) => {
            const cursor = (event.target as IDBRequest<IDBCursorWithValue>)
                .result
            if (cursor) {
                items.push(storedToHistoryItem(cursor.value as StoredHistoryItem))
                cursor.continue()
            } else {
                resolve(items)
            }
        }
    })
}

/**
 * Get a single history item by ID.
 *
 * @param id - Unique identifier for the history item
 * @returns Promise resolving to the history item with Object URLs, or null if not found
 */
export async function getHistoryItem(id: string): Promise<HistoryItem | null> {
    const db = getDB()

    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly")
        const store = tx.objectStore(STORE_NAME)
        const request = store.get(id)

        request.onerror = () => reject(request.error)
        request.onsuccess = () => {
            const stored = request.result as StoredHistoryItem | undefined
            resolve(stored ? storedToHistoryItem(stored) : null)
        }
    })
}

/**
 * Delete a history item by ID.
 * Automatically cleans up the stored images.
 *
 * @param id - Unique identifier for the history item to delete
 * @returns Promise that resolves when deletion is complete
 */
export async function deleteHistoryItem(id: string): Promise<void> {
    const db = getDB()

    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite")
        const store = tx.objectStore(STORE_NAME)
        const request = store.delete(id)

        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
    })
}

/**
 * Clear all history items from storage.
 *
 * @returns Promise that resolves when all items are deleted
 */
export async function clearHistory(): Promise<void> {
    const db = getDB()

    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite")
        const store = tx.objectStore(STORE_NAME)
        const request = store.clear()

        request.onerror = () => reject(request.error)
        request.onsuccess = () => resolve()
    })
}

/**
 * Convert a stored history item to a usable HistoryItem.
 * Converts ArrayBuffers back to Blobs and creates Object URLs for <img> tags.
 *
 * @param stored - Item from IndexedDB storage
 * @returns History item with Object URLs
 */
function storedToHistoryItem(stored: StoredHistoryItem): HistoryItem {
    // Convert ArrayBuffers back to Blobs
    const originalBlob = new Blob([stored.originalImage], {
        type: stored.originalImageType,
    })
    const processedBlob = new Blob([stored.processedImage], {
        type: stored.processedImageType,
    })

    return {
        id: stored.id,
        timestamp: stored.timestamp,
        originalImageUrl: URL.createObjectURL(originalBlob),
        processedImageUrl: URL.createObjectURL(processedBlob),
        results: stored.results,
    }
}

/**
 * Revoke Object URLs to prevent memory leaks.
 * Call this when a HistoryItem is no longer needed (e.g., component unmount).
 *
 * @param item - History item with Object URLs to revoke
 */
export function revokeHistoryItemUrls(item: HistoryItem): void {
    URL.revokeObjectURL(item.originalImageUrl)
    URL.revokeObjectURL(item.processedImageUrl)
}
