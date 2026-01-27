import React from "react"
import { Pressable, View } from "react-native"
import type { Detection } from "@/lib/model/types"
import { getDiseaseInfo } from "@/lib/model/disease-info"
import { Text } from "@/components/ui/text"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export type DetectionResultCardProps = {
    detection: Detection
    onPress: () => void
}

/**
 * DetectionResultCard - Shows one detection with disease name and confidence
 *
 * Features:
 * - Displays disease display name
 * - Shows confidence as percentage
 * - Tappable to navigate to disease info screen
 *
 * Requirements: 3.2, 6.3
 */
export function DetectionResultCard({
    detection,
    onPress,
}: DetectionResultCardProps) {
    const diseaseInfo = getDiseaseInfo(detection.diseaseClass)
    const confidencePercent = Math.round(detection.confidence * 100)

    // Severity color mapping
    const getSeverityColor = () => {
        switch (diseaseInfo.severity) {
            case "low":
                return "text-yellow-600"
            case "medium":
                return "text-orange-600"
            case "high":
                return "text-red-600"
            default:
                return "text-green-600"
        }
    }

    const severityColor = getSeverityColor()

    return (
        <Pressable onPress={onPress} className="active:opacity-70">
            <Card className="mb-3">
                <CardHeader>
                    <CardTitle className={severityColor}>
                        {diseaseInfo.displayName}
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <View className="flex-row justify-between items-center">
                        <Text className="text-sm text-muted-foreground">
                            Confidence
                        </Text>
                        <Text className="text-lg font-semibold">
                            {confidencePercent}%
                        </Text>
                    </View>
                    <Text className="text-xs text-muted-foreground mt-2">
                        Tap for more information
                    </Text>
                </CardContent>
            </Card>
        </Pressable>
    )
}
