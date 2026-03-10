import { baseConfig } from "../../js/eslint.config.mjs";

export default [
  ...baseConfig,
  {
    files: ["plugins/**/*.js"],
    languageOptions: {
      globals: {
        // Haraka SMTP result codes (injected into plugin scope at runtime)
        DENY: "readonly",
        DENYSOFT: "readonly",
        OK: "readonly",
        CONT: "readonly",
      },
    },
  },
];
