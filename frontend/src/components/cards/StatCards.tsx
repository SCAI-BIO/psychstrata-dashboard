import type { ReactNode } from "react";

export interface StatItem {
  label: string;
  value: ReactNode;
  /** Smaller inline suffix beside the value (e.g. "mg/day", "(3 Agents)"). */
  suffix?: string;
  caption: string;
}

/** The row of compact metric tiles at the bottom of the Medical view. */
export function StatCards({ items }: { items: StatItem[] }) {
  return (
    <section className="grid grid-cols-4 gap-4">
      {items.map((item) => (
        <article key={item.label} className="bg-white rounded-xl border border-slate-200/70 px-4 py-3.5">
          <p className="text-xs font-semibold text-slate-500 mb-1">{item.label}</p>
          <p className="text-2xl font-bold text-slate-900 leading-tight">
            {item.value}
            {item.suffix && <span className="text-sm font-medium text-slate-500 ml-1">{item.suffix}</span>}
          </p>
          <p className="text-xs text-slate-400 mt-1">{item.caption}</p>
        </article>
      ))}
    </section>
  );
}
