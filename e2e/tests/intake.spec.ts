import { expect, test } from "@playwright/test";
import { signIn } from "./helpers";

test.describe("patient intake", () => {
  test("runs the full intake wizard and produces a clinician prediction", async ({ page }) => {
    await page.goto("/intake");
    await signIn(page);

    await expect(page.getByRole("heading", { name: "New Patient" })).toBeVisible();

    // Step 1 — Demographics
    await page.getByPlaceholder("John").fill("Max");
    await page.getByPlaceholder("Doe").fill("E2E");
    await page.locator('input[type="date"]').fill("1985-04-12");
    await page.getByLabel("Gender at Birth").selectOption("Female");
    await page
      .getByLabel("Primary Diagnosis")
      .selectOption("F33.1 — Major depressive disorder, recurrent, moderate");
    await page.getByRole("button", { name: "Next Step" }).click();

    // Step 2 — Clinical Properties
    await expect(page.getByRole("heading", { name: "Clinical Information" })).toBeVisible();
    await page.getByPlaceholder("0-27").fill("18");
    await page.getByPlaceholder("Months").fill("24");
    await page.getByLabel("Sleep Disturbance").selectOption("Severe");
    await page.getByPlaceholder("0-200").fill("100");
    await page.getByPlaceholder("0-300").fill("25");
    await page.getByLabel("Side Effect Burden").selectOption("Moderate");
    await page.getByRole("button", { name: "Next Step" }).click();

    // Step 3 — Genetics (complete by default, then submit)
    await expect(page.getByRole("heading", { name: "Genomic Data" })).toBeVisible();
    await page.getByRole("button", { name: "Calculate Risk" }).click();

    // Clinician results
    await expect(page).toHaveURL(/\/results\/clinician/);
    await expect(page.getByText("Treatment Resistance Risk")).toBeVisible();
    await expect(page.getByText("Max E2E")).toBeVisible();
  });
});
