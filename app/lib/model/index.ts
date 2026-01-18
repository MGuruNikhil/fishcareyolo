/**
 * Model management module.
 *
 * Handles downloading, updating, and loading TFLite models from GitHub releases.
 */

// Types
export type {
    GitHubAsset,
    GitHubRelease,
    ModelChannel,
    ModelMetadata,
    ModelState,
    UpdateCheckResult,
} from "@/lib/model/types"

// Manager functions
export {
    checkForUpdate,
    downloadModel,
    forceUpdateModel,
    getLocalModelPath,
    getModelChannel,
    initializeModel,
    type DownloadProgressCallback,
} from "@/lib/model/manager"

// Storage utilities
export {
    clearModelMetadata,
    deleteModelFile,
    ensureModelDirectory,
    getModelDirectory,
    getModelFileSize,
    getModelPath,
    loadModelMetadata,
    modelFileExists,
    saveModelMetadata,
} from "@/lib/model/storage"

// React context
export { ModelProvider, useModel } from "@/lib/model/context"
