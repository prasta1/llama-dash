import type { ComponentType, ReactNode } from 'react'

type Props = {
  icon?: ComponentType<{ className?: string; 'aria-hidden'?: boolean }>
  children: ReactNode
  className?: string
}

export function EmptyState({ icon: Icon, children, className }: Props) {
  return (
    <div className={`empty-state empty-state-inviting flex flex-col items-center gap-2.5 py-9 ${className ?? ''}`}>
      {Icon ? (
        <span className="flex size-9 items-center justify-center rounded-full border border-border bg-surface-2 text-fg-faint">
          <Icon className="size-4" aria-hidden="true" />
        </span>
      ) : null}
      <div className="max-w-[42ch] text-center leading-[1.5]">{children}</div>
    </div>
  )
}
