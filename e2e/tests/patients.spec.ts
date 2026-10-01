import { expect, test } from "@playwright/test";
import { signIn } from "./helpers";

test.describe("patients", () => {
  test("lists the seeded demo patient and opens their results", async ({ page }) => {
    await page.goto("/");
    await signIn(page);

    const demoRow = page.getByRole("row", { name: /Max Mustermann/ });
    await expect(demoRow).toBeVisible();
    await expect(page.getByText(/in the database\./)).toBeVisible();

    await demoRow.getByRole("button", { name: "View Results" }).click();

    await expect(page).toHaveURL(/\/results\/clinician/);
    await expect(page.getByText("Treatment Resistance Risk")).toBeVisible();
  });
});
