import { Moon, ShieldCheck, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router'
import { cn } from '@/lib/utils'

type Theme = 'light' | 'dark'

function storedTheme(): Theme | null {
  try {
    const saved = localStorage.getItem('theme')
    return saved === 'light' || saved === 'dark' ? saved : null
  } catch {
    return null
  }
}

const systemTheme = (): Theme => (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')

export function useTheme() {
  const [chosen, setChosen] = useState<Theme | null>(storedTheme)
  const [system, setSystem] = useState<Theme>(systemTheme)
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-color-scheme: dark)')
    const onChange = () => setSystem(systemTheme())
    query?.addEventListener('change', onChange)
    return () => query?.removeEventListener('change', onChange)
  }, [])
  const theme = chosen ?? system
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])
  const choose = (next: Theme) => {
    setChosen(next)
    try {
      localStorage.setItem('theme', next)
    } catch {
      // ignore
    }
  }
  return [theme, choose] as const
}

function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <div className="flex size-7 items-center justify-center rounded-lg bg-emerald-600 text-white shadow-sm">
        <ShieldCheck className="size-4.5 stroke-[2.5]" />
      </div>
      <div className="flex flex-col">
        <span className="text-[1.05rem] font-black tracking-[-0.03em] text-ink">
          TRUTHLENS <span className="text-emerald-600 dark:text-emerald-400">AI</span>
        </span>
        <span className="hidden text-[0.68rem] font-medium tracking-tight text-ink-3 sm:inline">
          Fake News & Claim Verification
        </span>
      </div>
    </span>
  )
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'rounded-md px-3 py-1.5 text-sm font-semibold transition-colors duration-150',
    isActive ? 'bg-sunk text-ink shadow-xs' : 'text-ink-2 hover:text-ink',
  )

export function AppShell() {
  const [theme, setTheme] = useTheme()
  return (
    <div className="flex min-h-dvh flex-col bg-sheet text-ink">
      <header className="sticky top-0 z-30 border-b border-rule bg-paper/95 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1760px] items-center gap-6 px-4 sm:px-6">
          <NavLink to="/" aria-label="TruthLens AI Home" className="rounded-md">
            <Wordmark />
          </NavLink>
          <nav className="flex items-center gap-1.5" aria-label="Main Navigation">
            <NavLink to="/live" className={navClass}>
              Verify News
            </NavLink>
            <NavLink to="/runs" className={navClass}>
              History
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="inline-flex size-8 items-center justify-center rounded-md border border-rule text-ink-2 transition-[transform,color] duration-150 ease-out hover:text-ink active:scale-[0.96]"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
            >
              {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </div>
      </header>
      <main className="flex flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  )
}
