export type { HistoryItem, HistoryItemInput, StoredHistoryItem } from "./types"

export {
  clearHistory,
  closeHistoryDB,
  deleteHistoryItem,
  getHistoryItem,
  getHistoryItems,
  initHistoryDB,
  revokeHistoryItemUrls,
  saveHistoryItem,
} from "./storage"
