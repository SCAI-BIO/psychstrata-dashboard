import { type Page, expect } from "@playwright/test";

/** Credentials the compose stack is started with in CI (see system-tests.yaml). */
export const E2E_AUTH_USERNAME = process.env.E2E_AUTH_USERNAME ?? "e2e-user";
export const E2E_AUTH_PASSWORD = process.env.E2E_AUTH_PASSWORD ?? "e2e-pass";

/**
 * Sign in through the Basic-auth gate. Assumes the login form is (or becomes)
 * visible on the current page and waits until it has been replaced by the app.
 */
export async function signIn(
  page: Page,
  { username = E2E_AUTH_USERNAME, password = E2E_AUTH_PASSWORD }: { username?: string; password?: string } = {}
): Promise<void> {
  await expect(page.getByRole("heading", { name: "Dashboard login" })).toBeVisible();
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Dashboard login" })).toBeHidden();
}
