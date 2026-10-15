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

  it("ignores blank lines and surrounding whitespace", () => {
    render(<div data-testid="out">{renderSummaryMarkdown("\n\n   \n")}</div>);
    const out = screen.getByTestId("out");
    expect(out.querySelectorAll("p")).toHaveLength(0);
    expect(out.querySelectorAll("h4")).toHaveLength(0);
  });
});
