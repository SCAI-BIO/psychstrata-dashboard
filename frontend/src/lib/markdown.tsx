import { Fragment, type ReactNode } from "react";

/** Renders **bold** spans inside a single line of text. */
export function renderInlineMarkdown(text: string): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    const match = part.match(/^\*\*([^*]+)\*\*$/);
    if (match) {
      return <strong key={`strong-${index}`}>{match[1]}</strong>;
    }
    return <Fragment key={`text-${index}`}>{part}</Fragment>;
  });
}

/**
 * Minimal block-level markdown renderer for the LLM explanation:
 * supports `**heading**`, `- bullet` lists, and paragraphs.
 */
export function renderSummaryMarkdown(summary: string): ReactNode {
  const lines = summary.split("\n");
  const blocks: ReactNode[] = [];
  let listItems: ReactNode[] = [];
  let listKey = 0;

  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push(
        <ul key={`list-${listKey}`} className="space-y-1.5 text-sm text-slate-700 list-none pl-0 mt-2">
          {listItems}
        </ul>
      );
      listItems = [];
      listKey += 1;
    }
  };

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      return;
    }

    if (line.startsWith("- ")) {
      listItems.push(<li key={`item-${index}`}>{renderInlineMarkdown(line.slice(2))}</li>);
      return;
    }

    flushList();
    const headingMatch = line.match(/^\*\*(.+?)\*\*:?\s*$/);
    if (headingMatch) {
      blocks.push(
        <h4 key={`heading-${index}`} className="text-sm font-semibold text-slate-900 mt-4 mb-1">
          {headingMatch[1]}
        </h4>
      );
      return;
    }

    blocks.push(
      <p key={`paragraph-${index}`} className="text-sm text-slate-700 leading-relaxed">
        {renderInlineMarkdown(line)}
      </p>
    );
  });

  flushList();
  return <div className="text-sm text-slate-700 leading-relaxed space-y-2">{blocks}</div>;
}
