import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PatientProvider } from "../context/PatientContext";
import { ThemeProvider } from "../context/ThemeContext";
import { makePatientApi, makeCompletePatient } from "../test/fixtures";
import { AppShell } from "./AppShell";

vi.mock("./VersionBadge", () => ({ VersionBadge: () => <span>v-test</span> }));

function renderShell(props: Partial<Parameters<typeof AppShell>[0]> = {}) {
  const patientApi = makePatientApi(makeCompletePatient());
  return render(
    <ThemeProvider>
      <PatientProvider value={patientApi}>
        <AppShell role="clinician" onNavigate={vi.fn()} {...props}>
          <p>Body content</p>
        </AppShell>
      </PatientProvider>
    </ThemeProvider>
  );
}

describe("AppShell", () => {
  afterEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
  });

  it("renders the patient name, DOB and children", () => {
    renderShell();
    expect(screen.getByText("Body content")).toBeInTheDocument();
    expect(screen.getByText("John Doe")).toBeInTheDocument();
    expect(screen.getByText("DOB: 1978-05-12")).toBeInTheDocument();
  });

  it("navigates when a nav item is clicked", () => {
    const onNavigate = vi.fn();
    renderShell({ onNavigate });
    fireEvent.click(screen.getByRole("button", { name: "Scientific View" }));
    expect(onNavigate).toHaveBeenCalledWith("scientist");
  });

  it("navigates to the patient list and intake from the sidebar actions", () => {
    const onNavigate = vi.fn();
    renderShell({ onNavigate });
    fireEvent.click(screen.getByRole("button", { name: /Patient list/ }));
    expect(onNavigate).toHaveBeenCalledWith("patients");
    fireEvent.click(screen.getByRole("button", { name: /Add New Patient/ }));
    expect(onNavigate).toHaveBeenCalledWith("intake");
  });

  it("only renders the sign-out control when onLogout is supplied", () => {
    const { unmount } = renderShell();
    expect(screen.queryByRole("button", { name: /Sign out/ })).not.toBeInTheDocument();
    unmount();

    const onLogout = vi.fn();
    renderShell({ onLogout });
    fireEvent.click(screen.getByRole("button", { name: /Sign out/ }));
    expect(onLogout).toHaveBeenCalledTimes(1);
  });

  it("toggles the theme from the sidebar", () => {
    renderShell();
    const toggle = screen.getByRole("button", { name: "Toggle dark mode" });
    expect(toggle).toHaveTextContent("Dark mode");
    fireEvent.click(toggle);
    expect(screen.getByRole("button", { name: "Toggle dark mode" })).toHaveTextContent("Light mode");
    expect(document.documentElement).toHaveClass("dark");
  });
});
