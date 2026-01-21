import { MoonStarIcon, SunIcon } from "lucide-react-native"
import * as React from "react"
import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { useTheme } from "@/lib/theme"
import RootNavigator from "./navigation/RootNavigation"

export default function Screen() {
    return <RootNavigator />
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
