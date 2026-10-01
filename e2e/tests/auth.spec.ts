import { expect, test } from "@playwright/test";
import { E2E_AUTH_PASSWORD, E2E_AUTH_USERNAME, signIn } from "./helpers";

test.describe("authentication", () => {
  test("rejects invalid credentials with an error", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Username").fill(E2E_AUTH_USERNAME);
    await page.getByLabel("Password").fill("definitely-wrong");
    await page.getByRole("button", { name: "Sign in" }).click();

    await expect(page.getByText(/Invalid credentials/)).toBeVisible();
    await expect(page.getByRole("heading", { name: "Dashboard login" })).toBeVisible();
  });

  test("accepts valid credentials and lands on the patient list", async ({ page }) => {
    await page.goto("/");
    await signIn(page);

    await expect(page.getByRole("heading", { name: "Patients" })).toBeVisible();
  });
});
