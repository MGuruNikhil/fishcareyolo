import { useRouter } from "expo-router"
import { useEffect } from "react"

export function useDefaultTab() {
    const router = useRouter()

    useEffect(() => {
        // Redirect to index tab on app startup
        router.replace("/")
    }, [router])
}
