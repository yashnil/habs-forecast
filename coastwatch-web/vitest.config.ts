import { defineConfig } from "vitest/config";
import path from "node:path";

/**
 * CSS modules resolve to a stub whose class names are their own keys (`s.ticket === "ticket"`), so a unit test that
 * renders a design-system component never has to list every stylesheet the component imports (v14: the system grows
 * new ones). A test's own `vi.mock("….module.css")` still works.
 */
const STUB = "\0nx-css-module-stub";
const cssModuleStub = {
  name: "nx-css-module-stub",
  enforce: "pre" as const,
  resolveId(source: string) {
    return /\.module\.css$/.test(source) ? STUB : null;
  },
  load(id: string) {
    return id === STUB ? "export default new Proxy({}, { get: (_t, k) => (typeof k === 'string' ? k : undefined) });" : null;
  },
};

export default defineConfig({
  plugins: [cssModuleStub],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  test: { include: ["tests/unit/**/*.test.ts"], environment: "node" },
});
