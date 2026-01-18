import type { ExpoConfig, ConfigContext } from "expo/config"

export default ({ config }: ConfigContext): ExpoConfig => ({
    ...config,
    name: "mina",
    slug: "mina",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/images/icon.png",
    scheme: "mina",
    userInterfaceStyle: "automatic",
    newArchEnabled: true,
    splash: {
        image: "./assets/images/splash.png",
        resizeMode: "contain",
        backgroundColor: "#ffffff",
    },
    assetBundlePatterns: ["**/*"],
    ios: {
        supportsTablet: true,
        bundleIdentifier: "com.mina.app",
    },
    android: {
        edgeToEdgeEnabled: true,
        adaptiveIcon: {
            foregroundImage: "./assets/images/adaptive-icon.png",
            backgroundColor: "#ffffff",
        },
        package: "com.mina.app",
    },
    web: {
        bundler: "metro",
        output: "static",
        favicon: "./assets/images/favicon.png",
    },
    plugins: ["expo-router"],
    experiments: {
        typedRoutes: true,
    },
    extra: {
        // Model channel: "dev" or "prod"
        // Set via EXPO_PUBLIC_MODEL_CHANNEL env var at build time
        modelChannel: process.env.EXPO_PUBLIC_MODEL_CHANNEL ?? "prod",
    },
})
