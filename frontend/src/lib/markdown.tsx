import { Fragment, type ReactNode } from "react";

const PMID_CITATION = String.raw`PMIDs?\s*:?\s*\d{4,9}(?:\s*(?:,|;|and|&)\s*\d{4,9})*`;
const INLINE_PATTERN = new RegExp(
  String.raw`(\*\*[^*]+\*\*)|(\(\s*${PMID_CITATION}\s*\))|(\b${PMID_CITATION})`,
  "i"
);
const PMID_NUMBER_PATTERN = /\d{4,9}/g;

/** Sequential reference numbers assigned to PMIDs in order of appearance. */
type PmidRegistry = Map<string, number>;

function pmidHref(pmid: string): string {
  return `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
}

function referenceNumber(pmid: string, pmids: PmidRegistry): number {
  let reference = pmids.get(pmid);
  if (reference === undefined) {
    reference = pmids.size + 1;
    pmids.set(pmid, reference);
  }
  return reference;
}

/**
 * Renders **bold** spans and PMID citations inside a single line of text.
 * PMIDs become numbered `[n]` links to PubMed, numbered by order of appearance.
 */
export function renderInlineMarkdown(text: string, registry?: PmidRegistry): ReactNode {
  const pmids = registry ?? new Map<string, number>();
  const nodes: ReactNode[] = [];
  const pattern = new RegExp(INLINE_PATTERN.source, "gi");
  let lastIndex = 0;
  let key = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(<Fragment key={`text-${key++}`}>{text.slice(lastIndex, match.index)}</Fragment>);
    }

    if (match[1]) {
      nodes.push(<strong key={`strong-${key++}`}>{match[1].slice(2, -2)}</strong>);
      lastIndex = pattern.lastIndex;
      continue;
    }

    const citation = match[2] ?? match[3];
    const pmidsInCitation = citation.match(PMID_NUMBER_PATTERN) ?? [];

    nodes.push(
      <Fragment key={`pmids-${key++}`}>
        {pmidsInCitation.map((pmid, position) => (
          <Fragment key={`pmid-${key++}`}>
            {position > 0 ? ", " : ""}
            <a
              href={pmidHref(pmid)}
              target="_blank"
              rel="noreferrer"
              title={`View PMID ${pmid} on PubMed`}
              className="font-medium text-blue-700 dark:text-blue-300 hover:underline"
            >
              [{referenceNumber(pmid, pmids)}]
            </a>
          </Fragment>
        ))}
      </Fragment>
    );

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(<Fragment key={`text-${key++}`}>{text.slice(lastIndex)}</Fragment>);
  }

  return nodes;
}

/**
 * Minimal block-level markdown renderer for the LLM explanation:
 * supports `**heading**`, `- bullet` lists, and paragraphs.
 * PMID citations are rendered as numbered links to PubMed.
 */
export function renderSummaryMarkdown(summary: string): ReactNode {
  const lines = summary.split("\n");
  const blocks: ReactNode[] = [];
  const pmids: PmidRegistry = new Map();
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
      listItems.push(<li key={`item-${index}`}>{renderInlineMarkdown(line.slice(2), pmids)}</li>);
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
        {renderInlineMarkdown(line, pmids)}
      </p>
    );
  });

  flushList();
  return <div className="text-sm text-slate-700 leading-relaxed space-y-2">{blocks}</div>;
}
