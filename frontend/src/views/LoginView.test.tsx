import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeDashboard } from "../test/fixtures";
import { LoginView } from "./LoginView";

describe("LoginView", () => {
  it("renders the login sheet with the username prefilled", () => {
    render(<LoginView dashboard={makeDashboard()} />);
    expect(screen.getByRole("heading", { name: "Dashboard login" })).toBeInTheDocument();
    expect(screen.getByLabelText("Username")).toHaveValue("dashboard-user");
  });

  it("submits the entered credentials to attemptLogin", () => {
    const attemptLogin = vi.fn(async () => {});
    render(<LoginView dashboard={makeDashboard({ attemptLogin })} />);

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "clinician" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "s3cret" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(attemptLogin).toHaveBeenCalledWith("clinician", "s3cret");
  });

  it("shows the login error returned by the hook", () => {
    render(<LoginView dashboard={makeDashboard({ loginError: "Backend API error (401): Invalid credentials." })} />);
    expect(screen.getByText("Backend API error (401): Invalid credentials.")).toBeInTheDocument();
  });

  it("disables the form and shows progress while submitting", () => {
    render(<LoginView dashboard={makeDashboard({ isLoginSubmitting: true })} />);
    const button = screen.getByRole("button", { name: "Signing in…" });
    expect(button).toBeDisabled();
    expect(screen.getByLabelText("Username")).toBeDisabled();
  });
});
