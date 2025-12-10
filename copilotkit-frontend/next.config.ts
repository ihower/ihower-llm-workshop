import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Server external packages to avoid bundling issues
  serverExternalPackages: ["pino", "pino-pretty", "thread-stream"],
  // Transpile specific packages
  transpilePackages: ["@copilotkit/runtime"],
};

export default nextConfig;
