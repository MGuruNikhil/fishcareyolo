import React, { useEffect, useState } from "react"
import { Image, LayoutChangeEvent, View, ViewStyle } from "react-native"
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withDelay,
    withSpring,
} from "react-native-reanimated"
import type { Detection } from "@/lib/model/types"
import { getBoundingBoxColor } from "@/lib/model/disease-info"

export type DetectionOverlayProps = {
    imageUri: string
    detections: Detection[]
    animated?: boolean
}

/**
 * DetectionOverlay - Renders an image with bounding boxes overlaid
 *
 * Features:
 * - Displays image with proper aspect ratio
 * - Draws color-coded bounding boxes for each detection
 * - Animates boxes appearing progressively
 *
 * Requirements: 3.1, 7.4, 8.3
 */
export function DetectionOverlay({
    imageUri,
    detections,
    animated = true,
}: DetectionOverlayProps) {
    const [imageDimensions, setImageDimensions] = useState({
        width: 0,
        height: 0,
    })
    const [containerDimensions, setContainerDimensions] = useState({
        width: 0,
        height: 0,
    })

    // Get actual image dimensions
    useEffect(() => {
        Image.getSize(imageUri, (width, height) => {
            setImageDimensions({ width, height })
        })
    }, [imageUri])

    const handleLayout = (event: LayoutChangeEvent) => {
        const { width, height } = event.nativeEvent.layout
        setContainerDimensions({ width, height })
    }

    // Calculate how the image is displayed within the container
    const getDisplayDimensions = () => {
        if (
            !imageDimensions.width ||
            !imageDimensions.height ||
            !containerDimensions.width ||
            !containerDimensions.height
        ) {
            return null
        }

        const imageAspect = imageDimensions.width / imageDimensions.height
        const containerAspect =
            containerDimensions.width / containerDimensions.height

        let displayWidth: number
        let displayHeight: number
        let offsetX = 0
        let offsetY = 0

        if (imageAspect > containerAspect) {
            // Image is wider than container
            displayWidth = containerDimensions.width
            displayHeight = containerDimensions.width / imageAspect
            offsetY = (containerDimensions.height - displayHeight) / 2
        } else {
            // Image is taller than container
            displayHeight = containerDimensions.height
            displayWidth = containerDimensions.height * imageAspect
            offsetX = (containerDimensions.width - displayWidth) / 2
        }

        return { displayWidth, displayHeight, offsetX, offsetY }
    }

    const displayDimensions = getDisplayDimensions()

    return (
        <View className="relative w-full h-full" onLayout={handleLayout}>
            <Image
                source={{ uri: imageUri }}
                className="w-full h-full"
                resizeMode="contain"
            />

            {displayDimensions &&
                detections.map((detection, index) => (
                    <BoundingBox
                        key={detection.id}
                        detection={detection}
                        displayDimensions={displayDimensions}
                        animated={animated}
                        animationDelay={index * 100} // Stagger animation
                    />
                ))}
        </View>
    )
}

type BoundingBoxProps = {
    detection: Detection
    displayDimensions: {
        displayWidth: number
        displayHeight: number
        offsetX: number
        offsetY: number
    }
    animated: boolean
    animationDelay: number
}

function BoundingBox({
    detection,
    displayDimensions,
    animated,
    animationDelay,
}: BoundingBoxProps) {
    const scale = useSharedValue(animated ? 0 : 1)
    const opacity = useSharedValue(animated ? 0 : 1)

    useEffect(() => {
        if (animated) {
            scale.value = withDelay(
                animationDelay,
                withSpring(1, {
                    damping: 15,
                    stiffness: 150,
                }),
            )
            opacity.value = withDelay(
                animationDelay,
                withSpring(1, {
                    damping: 15,
                    stiffness: 150,
                }),
            )
        }
    }, [animated, animationDelay, scale, opacity])

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
        opacity: opacity.value,
    }))

    const { displayWidth, displayHeight, offsetX, offsetY } = displayDimensions
    const { boundingBox } = detection

    // Convert normalized coordinates to pixel coordinates
    const left = offsetX + boundingBox.x * displayWidth
    const top = offsetY + boundingBox.y * displayHeight
    const width = boundingBox.width * displayWidth
    const height = boundingBox.height * displayHeight

    const color = getBoundingBoxColor(detection.diseaseClass)

    const boxStyle: ViewStyle = {
        position: "absolute",
        left,
        top,
        width,
        height,
        borderWidth: 3,
        borderColor: color,
        borderRadius: 4,
    }

    return <Animated.View style={[boxStyle, animatedStyle]} />
}
