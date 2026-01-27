import { useLocalSearchParams, useRouter } from "expo-router"
import React, { useEffect, useState } from "react"
import { ActivityIndicator, ScrollView, View } from "react-native"
import { useNavigation } from "@react-navigation/native"
import { Text } from "@/components/ui/text"
import { Button } from "@/components/ui/button"
import { DetectionOverlay } from "@/components/ui/detection-overlay"
import { DetectionResultCard } from "@/components/ui/detection-result-card"
import { useModel } from "@/lib/model"
import { saveSession } from "@/lib/model"
import type { Detection, DetectionSession } from "@/lib/model/types"

/**
 * Results Screen - Displays disease detection results
 *
 * Features:
 * - Runs inference on mount
 * - Shows loading state during analysis
 * - Displays image with bounding box overlay
 * - Lists detection cards sorted by confidence
 * - Shows "healthy" message when no detections
 * - Saves session to storage on completion
 *
 * Requirements: 3.1, 3.2, 3.3, 3.4, 5.1
 */
export default function ResultsScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const { imageUri } = useLocalSearchParams<{ imageUri: string }>()
    const { runInference, isReady } = useModel()

    const [isLoading, setIsLoading] = useState(true)
    const [detections, setDetections] = useState<Detection[]>([])
    const [error, setError] = useState<string | null>(null)
    const [inferenceTime, setInferenceTime] = useState<number | null>(null)

    // Hide tab bar on this screen
    useEffect(() => {
        navigation.setOptions({
            tabBarStyle: { display: "none" },
        })

        return () => {
            navigation.setOptions({
                tabBarStyle: undefined,
            })
        }
    }, [navigation])

    // Run inference on mount
    useEffect(() => {
        const performInference = async () => {
            if (!imageUri) {
                setError("No image provided")
                setIsLoading(false)
                return
            }

            if (!isReady) {
                setError("Model not ready")
                setIsLoading(false)
                return
            }

            try {
                setIsLoading(true)
                setError(null)

                const result = await runInference(imageUri)

                if (!result) {
                    setError("Inference failed")
                    setIsLoading(false)
                    return
                }

                setDetections(result.detections)
                setInferenceTime(result.inferenceTimeMs)

                // Save session to storage (Requirement 5.1)
                const session: DetectionSession = {
                    id: `session_${Date.now()}`,
                    imageUri,
                    detections: result.detections,
                    timestamp: Date.now(),
                }

                await saveSession(session)
            } catch (err) {
                console.error("Inference error:", err)
                setError(
                    err instanceof Error
                        ? err.message
                        : "Unknown error occurred",
                )
            } finally {
                setIsLoading(false)
            }
        }

        performInference()
    }, [imageUri, isReady, runInference])

    const handleDetectionPress = (detection: Detection) => {
        router.push({
            pathname: "/disease-info",
            params: { diseaseClass: detection.diseaseClass },
        })
    }

    const handleRetry = () => {
        router.back()
    }

    const handleDone = () => {
        router.push("/")
    }

    if (!imageUri) {
        return (
            <View className="flex-1 bg-background items-center justify-center p-4">
                <Text className="text-destructive text-center mb-4">
                    No image provided
                </Text>
                <Button onPress={handleDone}>
                    <Text>Go Back</Text>
                </Button>
            </View>
        )
    }

    if (error) {
        return (
            <View className="flex-1 bg-background items-center justify-center p-4">
                <Text className="text-destructive text-center mb-4">
                    {error}
                </Text>
                <View className="flex-row gap-3">
                    <Button variant="outline" onPress={handleDone}>
                        <Text>Go Back</Text>
                    </Button>
                    <Button onPress={handleRetry}>
                        <Text>Retry</Text>
                    </Button>
                </View>
            </View>
        )
    }

    if (isLoading) {
        return (
            <View className="flex-1 bg-background items-center justify-center">
                <ActivityIndicator size="large" />
                <Text className="mt-4 text-muted-foreground">
                    Analyzing fish...
                </Text>
            </View>
        )
    }

    // Requirement 3.4: Show "healthy" message when no detections
    const hasDetections = detections.length > 0

    return (
        <View className="flex-1 bg-background">
            {/* Header */}
            <View className="px-4 pt-12 pb-4 border-b border-border">
                <Text className="text-2xl font-bold">Detection Results</Text>
                {inferenceTime !== null && (
                    <Text className="text-sm text-muted-foreground mt-1">
                        Analysis completed in {inferenceTime}ms
                    </Text>
                )}
            </View>

            <ScrollView className="flex-1">
                {/* Image with bounding boxes (Requirement 3.1) */}
                <View className="h-96 bg-black">
                    <DetectionOverlay
                        imageUri={imageUri}
                        detections={detections}
                        animated={true}
                    />
                </View>

                <View className="p-4">
                    {hasDetections ? (
                        <>
                            <Text className="text-lg font-semibold mb-3">
                                Detected Issues ({detections.length})
                            </Text>
                            {/* Requirement 3.2, 3.3: List detections sorted by confidence */}
                            {detections.map((detection) => (
                                <DetectionResultCard
                                    key={detection.id}
                                    detection={detection}
                                    onPress={() =>
                                        handleDetectionPress(detection)
                                    }
                                />
                            ))}
                        </>
                    ) : (
                        <View className="items-center justify-center py-8">
                            <Text className="text-lg text-green-600 font-semibold">
                                🐟 Your fish appears healthy!
                            </Text>
                            <Text className="text-sm text-muted-foreground mt-2 text-center">
                                No diseases detected in this image.
                            </Text>
                        </View>
                    )}
                </View>
            </ScrollView>

            {/* Bottom actions */}
            <View className="px-4 pb-8 pt-4 border-t border-border">
                <Button onPress={handleDone} className="w-full">
                    <Text>Done</Text>
                </Button>
            </View>
        </View>
    )
}
