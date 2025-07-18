"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Send, Bot, User } from "lucide-react"

export default function ModernChat() {
  const [messages, setMessages] = useState<any[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({
          behavior: "smooth",
          block: "end",
        })
      }
    }
    const timeoutId = setTimeout(scrollToBottom, 100)
    return () => clearTimeout(timeoutId)
  }, [messages, isLoading])

  // Scroll to bottom on initial load
  useEffect(() => {
    if (messagesEndRef.current && messages.length > 0) {
      messagesEndRef.current.scrollIntoView({ behavior: "auto" })
    }
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    const userMsg = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    }
    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setIsLoading(true)
    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: input }),
      })
      if (!res.ok) throw new Error("Network response was not ok")
      const data = await res.json()
      const replyMsg = {
        id: Date.now().toString() + "-bot",
        role: "assistant",
        content: data.answer || "Bot chưa trả lời!",
      }
      setMessages((prev) => [...prev, replyMsg])
    } catch (error) {
      const replyMsg = {
        id: Date.now().toString() + "-bot",
        role: "assistant",
        content: "Lỗi kết nối tới server!",
      }
      setMessages((prev) => [...prev, replyMsg])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl h-[90vh] flex flex-col bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm rounded-xl shadow-2xl border border-slate-200/50 dark:border-slate-700/50">
        {/* Header */}
        <div className="flex items-center gap-3 p-6 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
          <Avatar className="h-10 w-10 bg-gradient-to-r from-blue-500 to-purple-600">
            <AvatarFallback className="text-white">
              <Bot className="h-5 w-5" />
            </AvatarFallback>
          </Avatar>
          <div>
            <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100">UET AI</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">Trợ lý hỏi đáp về quy chế đào tạo</p>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full p-6" ref={scrollAreaRef}>
            <div className="space-y-6">
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <Bot className="h-12 w-12 mx-auto text-slate-400 mb-4" />
                  <h3 className="text-lg font-medium text-slate-600 dark:text-slate-300 mb-2">Chào bạn!</h3>
                  <p className="text-slate-500 dark:text-slate-400">
                    Tôi có thể giúp gì cho bạn
                  </p>
                </div>
              )}

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <Avatar className="h-8 w-8 bg-gradient-to-r from-blue-500 to-purple-600 flex-shrink-0">
                      <AvatarFallback className="text-white">
                        <Bot className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                  )}

                  <div
                    className={`max-w-[70%] rounded-2xl px-4 py-3 ${
                      message.role === "user"
                        ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white ml-auto"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>

                  {message.role === "user" && (
                    <Avatar className="h-8 w-8 bg-gradient-to-r from-green-500 to-emerald-600 flex-shrink-0">
                      <AvatarFallback className="text-white">
                        <User className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-3 justify-start">
                  <Avatar className="h-8 w-8 bg-gradient-to-r from-blue-500 to-purple-600 flex-shrink-0">
                    <AvatarFallback className="text-white">
                      <Bot className="h-4 w-4" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl px-4 py-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}

              {/* Add this div at the end for auto-scroll reference */}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        </div>

        {/* Input Area */}
        <div className="p-6 border-t border-slate-200 dark:border-slate-700 flex-shrink-0">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <Input
              value={input}
              onChange={handleInputChange}
              placeholder="Nhập câu hỏi của bạn"
              className="flex-1 rounded-full border-slate-300 dark:border-slate-600 focus:border-blue-500 dark:focus:border-blue-400 px-4 py-3 text-base"
              disabled={isLoading}
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="rounded-full h-12 w-12 p-0 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:opacity-50"
            >
              <Send className="h-5 w-5" />
            </Button>
          </form>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 text-center">Nhấn Enter để gửi</p>
        </div>
      </div>
    </div>
  )
}
