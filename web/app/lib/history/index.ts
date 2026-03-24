/**
 * History storage module.
 *
 * Provides IndexedDB-based persistent storage for detection history.
 */

// Types
export type { HistoryItem, HistoryItemInput, StoredHistoryItem } from "./types"

// Storage operations
export {
    clearHistory,
    deleteHistoryItem,
    getHistoryItem,
    getHistoryItems,
    initHistoryDB,
    revokeHistoryItemUrls,
    saveHistoryItem,
} from "./storage"
