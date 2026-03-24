import type { ComponentType } from "react"
import { BrowserRouter as Router } from "react-router-dom"
import { FileSystemRouter } from "file-system-router"

const pages = import.meta.glob("./pages/**/*.tsx", { eager: true }) as Record<
  string,
  { default: ComponentType }
>

export default function App() {
  return (
    <Router>
      <FileSystemRouter pages={pages} />
    </Router>
  )
}
