import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderInlineMarkdown, renderSummaryMarkdown } from "./markdown";

describe("renderInlineMarkdown", () => {
  it("renders plain text as a fragment", () => {
    render(<p data-testid="out">{renderInlineMarkdown("hello world")}</p>);
    expect(screen.getByTestId("out")).toHaveTextContent("hello world");
    expect(screen.getByTestId("out").querySelector("strong")).toBeNull();
  });

  it("wraps **bold** spans in <strong>", () => {
    render(<p data-testid="out">{renderInlineMarkdown("a **bold** move")}</p>);
    const strong = screen.getByTestId("out").querySelector("strong");
    expect(strong).toHaveTextContent("bold");
    expect(screen.getByTestId("out")).toHaveTextContent("a bold move");
  });

  it("handles multiple bold spans", () => {
    render(<p data-testid="out">{renderInlineMarkdown("**one** and **two**")}</p>);
    const strongs = screen.getByTestId("out").querySelectorAll("strong");
    expect(strongs).toHaveLength(2);
    expect(strongs[0]).toHaveTextContent("one");
    expect(strongs[1]).toHaveTextContent("two");
  });

  it("renders a PMID as a numbered PubMed link", () => {
    render(<p data-testid="out">{renderInlineMarkdown("shown (PMID: 17685743)")}</p>);
    const link = screen.getByTestId("out").querySelector("a");
    expect(link).toHaveTextContent("[1]");
    expect(link).toHaveAttribute("href", "https://pubmed.ncbi.nlm.nih.gov/17685743/");
    expect(link).toHaveAttribute("target", "_blank");
    expect(screen.getByTestId("out")).toHaveTextContent("shown [1]");
  });

  it("drops the surrounding parentheses from a parenthesized citation", () => {
    render(<p data-testid="out">{renderInlineMarkdown("shown (PMID: 17685743)")}</p>);
    expect(screen.getByTestId("out")).not.toHaveTextContent("(");
    expect(screen.getByTestId("out")).not.toHaveTextContent(")");
  });

  it("renders 'PMID:' without a space and without parentheses", () => {
    render(<p data-testid="out">{renderInlineMarkdown("see PMID:17685743 here")}</p>);
    const link = screen.getByTestId("out").querySelector("a");
    expect(link).toHaveTextContent("[1]");
    expect(screen.getByTestId("out")).toHaveTextContent("see [1] here");
  });

  it("renders a plural 'PMIDs:' list as multiple numbered links", () => {
    render(
      <p data-testid="out">
        {renderInlineMarkdown("difficult-to-treat depression (PMIDs: 34986373, 35993319)")}
      </p>
    );
    const links = screen.getByTestId("out").querySelectorAll("a");
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveTextContent("[1]");
    expect(links[0]).toHaveAttribute("href", "https://pubmed.ncbi.nlm.nih.gov/34986373/");
    expect(links[1]).toHaveTextContent("[2]");
    expect(links[1]).toHaveAttribute("href", "https://pubmed.ncbi.nlm.nih.gov/35993319/");
    expect(screen.getByTestId("out")).toHaveTextContent("difficult-to-treat depression [1], [2]");
  });

  it("renders a plural PMID list separated by 'and'", () => {
    render(<p data-testid="out">{renderInlineMarkdown("(PMIDs: 34986373 and 35993319)")}</p>);
    const links = screen.getByTestId("out").querySelectorAll("a");
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveTextContent("[1]");
    expect(links[1]).toHaveTextContent("[2]");
  });

  it("accepts PMID without a colon and reuses numbers for repeats", () => {
    render(<p data-testid="out">{renderInlineMarkdown("PMID 17685743 and PMID: 17685743")}</p>);
    const links = screen.getByTestId("out").querySelectorAll("a");
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveTextContent("[1]");
    expect(links[1]).toHaveTextContent("[1]");
  });

  it("numbers distinct PMIDs by order of appearance", () => {
    render(<p data-testid="out">{renderInlineMarkdown("PMID: 17444078 then PMID: 19192471")}</p>);
    const links = screen.getByTestId("out").querySelectorAll("a");
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveTextContent("[1]");
    expect(links[1]).toHaveTextContent("[2]");
  });
});

describe("renderSummaryMarkdown", () => {
  it("renders a **heading**: line as an <h4>", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("**Summary**")}</div>);
    const h4 = screen.getByTestId("out").querySelector("h4");
    expect(h4).toHaveTextContent("Summary");
  });

  it("renders headings with a trailing colon", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("**Key drivers**:")}</div>);
    expect(screen.getByTestId("out").querySelector("h4")).toHaveTextContent("Key drivers");
  });

  it("groups consecutive '- ' lines into a single <ul>", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("- first\n- second")}</div>);
    const uls = screen.getByTestId("out").querySelectorAll("ul");
    expect(uls).toHaveLength(1);
    expect(uls[0].querySelectorAll("li")).toHaveLength(2);
  });

  it("renders bullets with inline bold", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("- a **strong** point")}</div>);
    const li = screen.getByTestId("out").querySelector("li");
    expect(li?.querySelector("strong")).toHaveTextContent("strong");
  });

  it("renders plain lines as paragraphs", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("just a sentence")}</div>);
    expect(screen.getByTestId("out").querySelector("p")).toHaveTextContent("just a sentence");
  });

  it("flushes the list before a following paragraph", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("- one\nafter")}</div>);
    const out = screen.getByTestId("out");
    expect(out.querySelectorAll("ul")).toHaveLength(1);
    expect(out.querySelectorAll("p")).toHaveLength(1);
  });

  it("separates blocks split by blank lines", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("first\n\nsecond")}</div>);
    expect(screen.getByTestId("out").querySelectorAll("p")).toHaveLength(2);
  });

  it("numbers PMID citations continuously across blocks", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("**Summary**\n\n- PHQ-9 high (PMID: 17685743)\n- Duration long (PMID: 19192471)")}</div>);
    const links = screen.getByTestId("out").querySelectorAll("a");
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveTextContent("[1]");
    expect(links[1]).toHaveTextContent("[2]");
  });

  it("ignores blank lines and surrounding whitespace", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("\n\n   \n")}</div>);
    const out = screen.getByTestId("out");
    expect(out.querySelectorAll("p")).toHaveLength(0);
    expect(out.querySelectorAll("h4")).toHaveLength(0);
  });
});
