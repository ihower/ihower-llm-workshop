"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";

export default function Home() {
  return (
    <main className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">ihower + CopilotKit + AG-UI Demo</h1>


        <div className="bg-gray-100 p-6 rounded-lg">

        </div>
      </div>

      <CopilotSidebar
        labels={{
          title: "AI Assistant",
          initial: "你好！我是 AI 助理，有什麼可以幫助你的嗎？",
        }}
      />
    </main>
  );
}
