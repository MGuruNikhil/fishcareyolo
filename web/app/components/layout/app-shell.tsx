import { Outlet, NavLink } from "react-router-dom"
import { Camera, Clock, Settings } from "lucide-react"
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupContent,
    SidebarHeader,
    SidebarInset,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarProvider,
} from "@/components/ui/sidebar"
import { MobileNav } from "./mobile-nav"

const NAV_ITEMS = [
    { to: "/", label: "Camera", icon: Camera, end: true },
    { to: "/history", label: "History", icon: Clock },
    { to: "/settings", label: "Settings", icon: Settings },
]

export function AppShell() {
    return (
        <SidebarProvider>
            <div className="flex min-h-screen w-full">
                {/* Desktop Sidebar */}
                <Sidebar>
                    <SidebarHeader className="border-b p-5">
                        <div className="flex items-center gap-3">
                            <div className="flex size-8 items-center justify-center rounded-md bg-primary font-mono text-lg font-semibold text-primary-foreground">
                                M
                            </div>
                            <span className="font-mono text-lg font-semibold tracking-wide">
                                Mina
                            </span>
                        </div>
                    </SidebarHeader>
                    <SidebarContent>
                        <SidebarGroup>
                            <SidebarGroupContent>
                                <SidebarMenu>
                                    {NAV_ITEMS.map(
                                        ({ to, label, icon: Icon, end }) => (
                                            <SidebarMenuItem key={to}>
                                                <NavLink to={to} end={end}>
                                                    {({
                                                        isActive,
                                                    }: {
                                                        isActive: boolean
                                                    }) => (
                                                        <SidebarMenuButton
                                                            isActive={isActive}
                                                            className="min-h-11"
                                                        >
                                                            <Icon
                                                                size={18}
                                                                aria-hidden="true"
                                                            />
                                                            <span>{label}</span>
                                                        </SidebarMenuButton>
                                                    )}
                                                </NavLink>
                                            </SidebarMenuItem>
                                        ),
                                    )}
                                </SidebarMenu>
                            </SidebarGroupContent>
                        </SidebarGroup>
                    </SidebarContent>
                    <SidebarFooter className="border-t p-4">
                        <span className="font-mono text-xs text-muted-foreground">
                            v1.0.0
                        </span>
                    </SidebarFooter>
                </Sidebar>

                {/* Main Content */}
                <SidebarInset>
                    <main
                        className="flex min-h-screen flex-1 flex-col pb-[60px] md:pb-0"
                        id="main-content"
                    >
                        <Outlet />
                    </main>
                </SidebarInset>

                {/* Mobile Bottom Nav */}
                <MobileNav />
            </div>
        </SidebarProvider>
    )
}
