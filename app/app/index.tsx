import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"
import { useTheme } from "@/lib/theme-context"
import { Link, Stack } from "expo-router"
import { FishIcon, MoonStarIcon, StarIcon, SunIcon } from "lucide-react-native"
import * as React from "react"
import { View } from "react-native"

const SCREEN_OPTIONS = {
    title: "mina",
    headerTransparent: true,
    headerRight: () => <ThemeToggle />,
}

export default function Screen() {
    const { colorScheme } = useTheme()

    return (
        <>
            <Stack.Screen options={SCREEN_OPTIONS} />
            <View className="flex-1 items-center justify-center gap-8 p-4">
                <View className="flex-row items-center gap-4">
                    <Icon as={FishIcon} className="h-16 w-12" />
                </View>
                <View className="gap-2 p-4">
                    <Text className="ios:text-foreground font-mono text-sm text-muted-foreground">
                        1. Edit <Text variant="code">app/index.tsx</Text> to get
                        started.
                    </Text>
                    <Text className="ios:text-foreground font-mono text-sm text-muted-foreground">
                        2. Save to see your changes instantly.
                    </Text>
                </View>
            </View>
        </>
    )
}

const THEME_ICONS = {
    light: SunIcon,
    dark: MoonStarIcon,
}

function ThemeToggle() {
    const { colorScheme, toggleColorScheme } = useTheme()

    return (
        <Button
            onPressIn={toggleColorScheme}
            size="icon"
            variant="ghost"
            className="ios:size-9 rounded-full web:mx-4"
        >
            <Icon as={THEME_ICONS[colorScheme]} className="size-5" />
        </Button>
    )
}
