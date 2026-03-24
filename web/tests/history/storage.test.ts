import { describe, expect, it } from "bun:test"
// @ts-ignore - fake-indexeddb is missing types
import fakeIndexedDB from "fake-indexeddb"
// @ts-ignore - fake-indexeddb is missing types
import FDBKeyRange from "fake-indexeddb/lib/FDBKeyRange"

// Polyfill IndexedDB for Bun
globalThis.indexedDB = fakeIndexedDB
globalThis.IDBKeyRange = FDBKeyRange

import type { HistoryItemInput } from "@/lib/history/types"
import {
    clearHistory,
    deleteHistoryItem,
    getHistoryItem,
    getHistoryItems,
    initHistoryDB,
    saveHistoryItem,
} from "@/lib/history/storage"

// Helper function to create test data
async function createTestHistoryInput(
    timestamp: number,
): Promise<HistoryItemInput> {
    // Use simple Uint8Array data that's easier to serialize
    const originalData = new Uint8Array([1, 2, 3, 4, 5])
    const processedData = new Uint8Array([6, 7, 8, 9, 10])

    return {
        timestamp,
        originalImage: new Blob([originalData], { type: "image/png" }),
        processedImage: new Blob([processedData], { type: "image/png" }),
        results: {
            detections: [
                {
                    id: "det_001",
                    diseaseClass: "bacterial_infection",
                    confidence: 0.85,
                    boundingBox: { x: 0.1, y: 0.1, width: 0.3, height: 0.3 },
                },
            ],
            inferenceTimeMs: 150,
        },
    }
}

// Clean up database
async function cleanupDB() {
    try {
        await clearHistory()
    } catch {
        // Ignore errors during cleanup
    }
}

describe("**Feature: fish-disease-detection, HistoryStorage**", () => {
    describe("initHistoryDB", () => {
        it("should initialize database successfully", async () => {
            await initHistoryDB()
            // If we get here, initialization succeeded
        })
    })

    describe("saveHistoryItem and getHistoryItems", () => {
        it("should save and retrieve a history item", async () => {
            await initHistoryDB()
            await cleanupDB()

            const input = await createTestHistoryInput(Date.now())
            const saved = await saveHistoryItem(input)

            expect(saved.id).toBeDefined()
            expect(typeof saved.id).toBe("string")

            const items = await getHistoryItems()
            expect(items).toHaveLength(1)
            expect(items[0].id).toBe(saved.id)

            await cleanupDB()
        })

        it("should save multiple items", async () => {
            await initHistoryDB()
            await cleanupDB()

            await saveHistoryItem(await createTestHistoryInput(1000))
            await saveHistoryItem(await createTestHistoryInput(2000))
            await saveHistoryItem(await createTestHistoryInput(3000))

            const items = await getHistoryItems()
            expect(items).toHaveLength(3)

            await cleanupDB()
        })

        it("should return empty array when no items exist", async () => {
            await initHistoryDB()
            await cleanupDB()

            const items = await getHistoryItems()
            expect(items).toEqual([])
        })
    })

    describe("**Feature: fish-disease-detection, Property 4: History sorting by timestamp**", () => {
        it("should return items sorted by timestamp descending", async () => {
            await initHistoryDB()
            await cleanupDB()

            await saveHistoryItem(await createTestHistoryInput(1000))
            await saveHistoryItem(await createTestHistoryInput(3000))
            await saveHistoryItem(await createTestHistoryInput(2000))

            const items = await getHistoryItems()

            expect(items).toHaveLength(3)
            expect(items[0].timestamp).toBe(3000)
            expect(items[1].timestamp).toBe(2000)
            expect(items[2].timestamp).toBe(1000)

            await cleanupDB()
        })

        it("should handle items saved in random order", async () => {
            await initHistoryDB()
            await cleanupDB()

            const timestamps = [5000, 1000, 3000, 4000, 2000]
            for (const ts of timestamps) {
                await saveHistoryItem(await createTestHistoryInput(ts))
            }

            const items = await getHistoryItems()

            expect(items).toHaveLength(5)
            for (let i = 0; i < items.length - 1; i++) {
                expect(items[i].timestamp).toBeGreaterThanOrEqual(
                    items[i + 1].timestamp,
                )
            }

            await cleanupDB()
        })
    })

    describe("getHistoryItem", () => {
        it("should return null for non-existent ID", async () => {
            await initHistoryDB()
            await cleanupDB()

            const result = await getHistoryItem("non_existent_id")
            expect(result).toBeNull()
        })

        it("should retrieve exact item that was saved", async () => {
            await initHistoryDB()
            await cleanupDB()

            const input = await createTestHistoryInput(12345)
            const saved = await saveHistoryItem(input)

            const retrieved = await getHistoryItem(saved.id)

            expect(retrieved).not.toBeNull()
            expect(retrieved?.id).toBe(saved.id)
            expect(retrieved?.timestamp).toBe(12345)
            expect(retrieved?.results.inferenceTimeMs).toBe(150)

            await cleanupDB()
        })
    })

    describe("deleteHistoryItem", () => {
        it("should remove item from storage", async () => {
            await initHistoryDB()
            await cleanupDB()

            const saved = await saveHistoryItem(
                await createTestHistoryInput(Date.now()),
            )
            expect(await getHistoryItem(saved.id)).not.toBeNull()

            await deleteHistoryItem(saved.id)

            expect(await getHistoryItem(saved.id)).toBeNull()

            await cleanupDB()
        })

        it("should not affect other items", async () => {
            await initHistoryDB()
            await cleanupDB()

            const item1 = await saveHistoryItem(await createTestHistoryInput(1000))
            const item2 = await saveHistoryItem(await createTestHistoryInput(2000))
            const item3 = await saveHistoryItem(await createTestHistoryInput(3000))

            await deleteHistoryItem(item2.id)

            const items = await getHistoryItems()
            expect(items).toHaveLength(2)

            const ids = items.map((item) => item.id).sort()
            expect(ids).toEqual([item1.id, item3.id].sort())

            await cleanupDB()
        })
    })

    describe("clearHistory", () => {
        it("should remove all items from storage", async () => {
            await initHistoryDB()
            await cleanupDB()

            await saveHistoryItem(await createTestHistoryInput(1000))
            await saveHistoryItem(await createTestHistoryInput(2000))
            await saveHistoryItem(await createTestHistoryInput(3000))

            let items = await getHistoryItems()
            expect(items).toHaveLength(3)

            await clearHistory()

            items = await getHistoryItems()
            expect(items).toHaveLength(0)
        })
    })
})
