import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface CardProps {
  /** Optional lucide icon rendered before the title. */
  icon?: LucideIcon;
  title?: ReactNode;
  /** Right-aligned slot in the header row (button, badge, pill…). */
  action?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}

/**
 * The white rounded panel used everywhere in the result views. Centralises the
 * border / shadow / padding so every card reads identically to the mockups.
 */
export function Card({ icon: Icon, title, action, className = "", bodyClassName = "", children }: CardProps) {
  const hasHeader = Boolean(title) || Boolean(action);
  return (
    <article className={`bg-white rounded-xl border border-slate-200/80 shadow-sm p-6 ${className}`}>
      {hasHeader && (
        <header className="flex items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2 text-slate-900">
            {Icon && <Icon size={18} className="text-slate-500" />}
            {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
          </div>
          {action}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </article>
  );
}

/** Small uppercase section label used inside cards (e.g. "Clinical Factors"). */
export function SectionLabel({ children }: { children: ReactNode }) {
  return <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">{children}</p>;
}
