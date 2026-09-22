import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { VersionBadge } from "./VersionBadge";

describe("VersionBadge", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the backend version after loading", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ version: "1.2.3" }), { status: 200 })
    ));

    render(<VersionBadge />);
    expect(screen.getByText("Loading version…")).toBeInTheDocument();
    expect(await screen.findByText("Version 1.2.3")).toBeInTheDocument();
  });

  it("renders an explicit fallback when the backend is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    render(<VersionBadge />);
    await waitFor(() => expect(screen.getByText("Version unavailable")).toBeInTheDocument());
  });
});
