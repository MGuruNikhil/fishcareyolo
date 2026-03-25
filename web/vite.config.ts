import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { VitePWA } from "vite-plugin-pwa"

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.ico", "apple-touch-icon.png", "masked-icon.svg", "*.wasm"],
      manifest: {
        name: "mina",
        short_name: "mina",
        description: "fish disease detection using yolo model",
        theme_color: "#ffffff",
        icons: [
          {
            src: "pwa-192x192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
          },
        ],
      },
      workbox: {
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2,wasm}"],
        runtimeCaching: [
          {
            urlPattern:
              /^https:\/\/github\.com\/fishcareyolo\/fishcareyolo\/releases\/download\/prod\/.*\.onnx$/,
            handler: "CacheFirst",
            options: {
              cacheName: "onnx-models",
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365,
              },
              cacheableResponse: {
                statuses: [0, 200],
              },
            },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./app"),
    },
  },
})
