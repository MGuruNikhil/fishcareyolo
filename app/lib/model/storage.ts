/**
 * Local storage utilities for model files and metadata.
 */

import AsyncStorage from "@react-native-async-storage/async-storage"
import * as FileSystem from "expo-file-system"
import type { ModelChannel, ModelMetadata } from "@/lib/model/types"

/** Storage keys */
const STORAGE_KEYS = {
    METADATA: "mina_model_metadata",
} as const

/** Get the directory where models are stored */
export function getModelDirectory(): string {
    return `${FileSystem.documentDirectory}models/`
}

/** Get the path to the model file for a given channel */
export function getModelPath(channel: ModelChannel): string {
    return `${getModelDirectory()}${channel}_model.tflite`
}

/** Ensure the model directory exists */
export async function ensureModelDirectory(): Promise<void> {
    const dir = getModelDirectory()
    const info = await FileSystem.getInfoAsync(dir)
    if (!info.exists) {
        await FileSystem.makeDirectoryAsync(dir, { intermediates: true })
    }
}

/** Save model metadata to AsyncStorage */
export async function saveModelMetadata(
    metadata: ModelMetadata,
): Promise<void> {
    await AsyncStorage.setItem(STORAGE_KEYS.METADATA, JSON.stringify(metadata))
}

/** Load model metadata from AsyncStorage */
export async function loadModelMetadata(): Promise<ModelMetadata | null> {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.METADATA)
    if (!data) return null

    try {
        return JSON.parse(data) as ModelMetadata
    } catch {
        return null
    }
}

/** Clear model metadata from AsyncStorage */
export async function clearModelMetadata(): Promise<void> {
    await AsyncStorage.removeItem(STORAGE_KEYS.METADATA)
}

/** Check if a model file exists locally */
export async function modelFileExists(channel: ModelChannel): Promise<boolean> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    return info.exists
}

/** Delete the local model file */
export async function deleteModelFile(channel: ModelChannel): Promise<void> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    if (info.exists) {
        await FileSystem.deleteAsync(path)
    }
}

/** Get the size of the local model file in bytes */
export async function getModelFileSize(
    channel: ModelChannel,
): Promise<number | null> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    if (info.exists && "size" in info) {
        return info.size
    }
    return null
}
