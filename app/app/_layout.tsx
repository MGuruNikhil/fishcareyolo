import "@/global.css"

import { NAV_THEME } from "@/lib/theme"
import { ThemeProvider as NavThemeProvider } from "@react-navigation/native"
import { PortalHost } from "@rn-primitives/portal"
import { Stack } from "expo-router"
import { StatusBar } from "expo-status-bar"
import { ThemeProvider, useTheme } from "@/lib/theme-context"

export { ErrorBoundary } from "expo-router"

function RootLayoutContent() {
    const { colorScheme } = useTheme()

    return (
        <NavThemeProvider value={NAV_THEME[colorScheme]}>
            <StatusBar style={colorScheme === "dark" ? "light" : "dark"} />
            <Stack />
            <PortalHost />
        </NavThemeProvider>
    )
}

export default function RootLayout() {
    return (
        <ThemeProvider>
            <RootLayoutContent />
        </ThemeProvider>
    )
}
