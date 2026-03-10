import js from "@eslint/js";
import globals from "globals";

/** Shared ESLint base config for tobira JS projects. */
export const baseConfig = [
  js.configs.recommended,
  {
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
];

export default baseConfig;
