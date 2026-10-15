import { act, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ThemeProvider, useTheme } from "./ThemeContext";

const STORAGE_KEY = "psychstrata-theme";

function matchMediaMock(matches: boolean) {
  return vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn()
  }));
}

describe("ThemeContext", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("throws when useTheme is used outside a provider", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => useTheme())).toThrow(/within a ThemeProvider/);
    spy.mockRestore();
  });

  it("defaults to light when no stored value and the OS prefers light", () => {
    vi.stubGlobal("matchMedia", matchMediaMock(false));

    const { result } = renderHook(() => useTheme(), { wrapper: ThemeProvider });

    expect(result.current.theme).toBe("light");
  });

  it("defaults to dark when the OS prefers dark", () => {
    vi.stubGlobal("matchMedia", matchMediaMock(true));

    const { result } = renderHook(() => useTheme(), { wrapper: ThemeProvider });

    expect(result.current.theme).toBe("dark");
  });

  it("prefers a stored theme over the OS preference", () => {
    vi.stubGlobal("matchMedia", matchMediaMock(true));
    localStorage.setItem(STORAGE_KEY, "light");

    const { result } = renderHook(() => useTheme(), { wrapper: ThemeProvider });

    expect(result.current.theme).toBe("light");
  });

  it("ignores an invalid stored value and falls back to the OS preference", () => {
    vi.stubGlobal("matchMedia", matchMediaMock(false));
    localStorage.setItem(STORAGE_KEY, "banana");

    const { result } = renderHook(() => useTheme(), { wrapper: ThemeProvider });

    expect(result.current.theme).toBe("light");
  });

  it("toggles the theme, updates the class list, and persists it", () => {
    vi.stubGlobal("matchMedia", matchMediaMock(false));

    const { result } = renderHook(() => useTheme(), { wrapper: ThemeProvider });
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    act(() => result.current.toggleTheme());

    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(localStorage.getItem(STORAGE_KEY)).toBe("dark");

    act(() => result.current.toggleTheme());

    expect(result.current.theme).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(localStorage.getItem(STORAGE_KEY)).toBe("light");
  });

  it("renders children", () => {
    vi.stubGlobal("matchMedia", matchMediaMock(false));
    render(
      <ThemeProvider>
        <span>child-content</span>
      </ThemeProvider>
    );
    expect(screen.getByText("child-content")).toBeInTheDocument();
  });
});
