/**
 * Type definitions for model management.
 */

/** Model distribution channel */
export type ModelChannel = "dev" | "prod"

/** Metadata about a model release */
export interface ModelMetadata {
    /** Release channel (dev or prod) */
    channel: ModelChannel
    /** ISO date string when the release was updated */
    updatedAt: string
    /** Model file size in bytes */
    sizeBytes: number
    /** URL the model was downloaded from */
    downloadUrl: string
}

/** Current state of model management */
export interface ModelState {
    /** Whether the model is ready for inference */
    isReady: boolean
    /** Whether a download/update is in progress */
    isLoading: boolean
    /** Download progress (0-1) */
    progress: number
    /** Error message if something went wrong */
    error: string | null
    /** Metadata about the currently loaded model */
    metadata: ModelMetadata | null
}

/** Result of checking for model updates */
export interface UpdateCheckResult {
    /** Whether an update is available */
    hasUpdate: boolean
    /** The new release date if update available */
    newDate: string | null
    /** Current local release date */
    currentDate: string | null
}

/** GitHub release API response (simplified) */
export interface GitHubRelease {
    tag_name: string
    name: string
    body: string
    published_at: string
    assets: GitHubAsset[]
}

/** GitHub release asset */
export interface GitHubAsset {
    name: string
    browser_download_url: string
    size: number
}
