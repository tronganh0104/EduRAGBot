"use client"

import React, { useState } from "react"
import { cn } from "@/lib/utils"
import { ArrowUpIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { AutoResizeTextarea } from "@/components/autoresize-textarea"

// Thay YOUR_GEMINI_API_KEY bằng API key thật của bạn
const GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + GEMINI_API_KEY

export function ChatForm({ className, ...props }: React.ComponentProps<"form">) {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Xin chào! Tôi là Gemini. Bạn cần hỏi gì?" },
  ])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!input.trim() || loading) return
    const userMsg = { role: "user", content: input }
    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setLoading(true)
    try {
      const reply = await fetchGeminiReply([...messages, userMsg])
      setMessages((prev) => [...prev, { role: "assistant", content: reply }])
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: "Xin lỗi, tôi không thể trả lời lúc này." }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>)
    }
  }

  return (
    <div className={cn(
      "flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-50 to-gray-200 dark:from-gray-900 dark:to-gray-800 p-2",
      className
    )}>
      <div className="w-full max-w-md flex flex-col rounded-2xl shadow-xl bg-white/90 dark:bg-gray-900/90 border border-gray-200 dark:border-gray-800 h-[80vh] max-h-[700px] overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-3 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700">
          {messages.map((message, idx) => (
            <div
              key={idx}
              className={cn(
                "flex",
                message.role === "user" ? "justify-end" : "justify-start"
              )}
            >
              <div
                className={cn(
                  "px-4 py-2 rounded-2xl max-w-[75%] text-sm shadow-sm",
                  message.role === "user"
                    ? "bg-blue-500 text-white rounded-br-md"
                    : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-bl-md"
                )}
              >
                {message.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="px-4 py-2 rounded-2xl max-w-[75%] text-sm shadow-sm bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-bl-md opacity-70 animate-pulse">
                Gemini đang trả lời...
              </div>
            </div>
          )}
        </div>
        <form
          onSubmit={handleSubmit}
          className="relative flex items-center gap-2 p-4 border-t border-gray-100 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80"
        >
          <AutoResizeTextarea
            onKeyDown={handleKeyDown}
            onChange={setInput}
            value={input}
            placeholder="Nhập tin nhắn..."
            className="flex-1 resize-none rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 dark:focus:ring-blue-600 transition"
            disabled={loading}
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="submit"
                variant="ghost"
                size="sm"
                className="rounded-full p-2 hover:bg-blue-100 dark:hover:bg-blue-900 transition"
                disabled={!input.trim() || loading}
              >
                <ArrowUpIcon size={18} />
              </Button>
            </TooltipTrigger>
            <TooltipContent sideOffset={12}>Gửi</TooltipContent>
          </Tooltip>
        </form>
      </div>
    </div>
  )
}

// Hàm gọi API Gemini
async function fetchGeminiReply(messages: { role: string; content: string }[]): Promise<string> {
  // Chuyển đổi messages sang format của Gemini
  const history = messages.map((msg) => ({
    role: msg.role === "user" ? "user" : "model",
    parts: [{ text: msg.content }],
  }))
  const res = await fetch(GEMINI_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: history }),
  })
  if (!res.ok) throw new Error("Gemini API error")
  const data = await res.json()
  // Lấy text trả về đầu tiên
  return (
    data?.candidates?.[0]?.content?.parts?.[0]?.text ||
    "(Không có phản hồi từ Gemini)"
  )
} 