import { openDB, type DBSchema, type IDBPDatabase } from "idb"
import type { HistoryItem, StoredHistoryItem, HistoryItemInput } from "./types"
import { generateUUID } from "@/lib/utils/uuid"

const DB_NAME = "fishcare_history"
const DB_VERSION = 1
const STORE_NAME = "history_items"

interface HistoryDB extends DBSchema {
  [STORE_NAME]: {
    key: string
    value: StoredHistoryItem
    indexes: { timestamp: number }
  }
}

let dbInstance: IDBPDatabase<HistoryDB> | null = null

export async function initHistoryDB(): Promise<void> {
  if (dbInstance) return
  dbInstance = await openDB<HistoryDB>(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" })
        store.createIndex("timestamp", "timestamp")
      }
    },
  })
}

export function closeHistoryDB(): void {
  if (dbInstance) {
    dbInstance.close()
    dbInstance = null
  }
}

function getDB(): IDBPDatabase<HistoryDB> {
  if (!dbInstance) {
    throw new Error("Database not initialized. Call initHistoryDB() first.")
  }
  return dbInstance
}

export async function saveHistoryItem(input: HistoryItemInput): Promise<HistoryItem> {
  const db = getDB()
  const id = generateUUID()

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

  await db.add(STORE_NAME, stored)
  return storedToHistoryItem(stored)
}

export async function getHistoryItems(): Promise<HistoryItem[]> {
  const db = getDB()
  const stored = await db.getAllFromIndex(STORE_NAME, "timestamp")
  return stored.reverse().map(storedToHistoryItem)
}

export async function getHistoryItem(id: string): Promise<HistoryItem | null> {
  const db = getDB()
  const stored = await db.get(STORE_NAME, id)
  return stored ? storedToHistoryItem(stored) : null
}

export async function deleteHistoryItem(id: string): Promise<void> {
  const db = getDB()
  await db.delete(STORE_NAME, id)
}

export async function clearHistory(): Promise<void> {
  const db = getDB()
  await db.clear(STORE_NAME)
}

function storedToHistoryItem(stored: StoredHistoryItem): HistoryItem {
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

export function revokeHistoryItemUrls(item: HistoryItem): void {
  URL.revokeObjectURL(item.originalImageUrl)
  URL.revokeObjectURL(item.processedImageUrl)
}
