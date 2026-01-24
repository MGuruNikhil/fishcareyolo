import React from "react"
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs"
import { House, FileText, Settings as SettingsIcon } from "lucide-react-native"
import { useSafeAreaInsets } from "react-native-safe-area-context"
import HomeStack from "@/app/navigation/HomeStack"
import History from "@/app/screens/history/history"
import Settings from "@/app/screens/settings/settings"
import { BottomTabsList } from "@/app/navigation/BottomTabsList.types"

const Tab = createBottomTabNavigator<BottomTabsList>()

const TabBarIcon = ({ route, color }: { route: any; color: string }) => {
    if (route.name === "Home") {
        return <House size={24} color={color} />
    } else if (route.name === "History") {
        return <FileText size={24} color={color} />
    } else if (route.name === "Settings") {
        return <SettingsIcon size={24} color={color} />
    }

    //Default
    return <House size={24} color={color} />
}

// Function to create tabBarIcon outside component to avoid re-creation
const getTabBarIcon = (route: any) => {
    return ({ color }: { color: string }) => (
        <TabBarIcon route={route} color={color} />
    )
}

export default function BottomTabs() {
    const insets = useSafeAreaInsets()

    return (
        <Tab.Navigator
            screenOptions={({ route }) => ({
                tabBarIcon: getTabBarIcon(route),
                tabBarActiveTintColor: "#007AFF",
                tabBarInactiveTintColor: "#999999",
                tabBarShowLabel: false,
                headerShown: false,
                tabBarHideOnKeyboard: true,
                tabBarStyle: {
                    backgroundColor: "#FFFFFF",
                    height: 62 + insets.bottom,
                    paddingBottom: Math.max(5, insets.bottom),
                    paddingTop: 5,
                },
            })}
        >
            <Tab.Screen name="Home" component={HomeStack} />
            <Tab.Screen name="History" component={History} />
            <Tab.Screen name="Settings" component={Settings} />
        </Tab.Navigator>
    )
}
