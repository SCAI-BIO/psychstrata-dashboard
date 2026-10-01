import { expect, test } from "@playwright/test";

test.describe("app shell", () => {
  test("serves the SPA and the backend is healthy through the nginx proxy", async ({ page, request }) => {
    await page.goto("/");

    // The stack is started with Basic Auth enabled, so the app gates on login.
    await expect(page.getByText("PsychStrata CDSS")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Dashboard login" })).toBeVisible();

    const health = await request.get("/api/health");
    expect(health.ok()).toBeTruthy();
    expect(await health.json()).toEqual({ status: "ok" });

    const authStatus = await request.get("/api/auth/status");
    expect(authStatus.ok()).toBeTruthy();
    expect(await authStatus.json()).toEqual({ auth_enabled: true });
  });
});
