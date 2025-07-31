"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Send, Bot, User, Cpu } from "lucide-react"
import { BACKEND_URL } from "./config"

const AI_MODELS = [
  {
    id: "Qwen3 4B pretrain",
    name: "Qwen3 4B pretrain",
    description: "",
    provider: "",
    color: "bg-green-500",
  },
  {
    id: "Qwen3 1.7B",
    name: "Qwen3 1.7B",
    description: "",
    provider: "",
    color: "bg-blue-500",
  },
  {
    id: "Qwen3 4B finetune",
    name: "Qwen3 4B finetune",
    description: "",
    provider: "",
    color: "bg-red-500",
  },
]

export default function ModernChat() {
  const [messages, setMessages] = useState<any[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState(AI_MODELS[0].id)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
    }
    const timeoutId = setTimeout(scrollToBottom, 100)
    return () => clearTimeout(timeoutId)
  }, [messages, isLoading])

  useEffect(() => {
    if (messagesEndRef.current && messages.length > 0) {
      messagesEndRef.current.scrollIntoView({ behavior: "auto" })
    }
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => setInput(e.target.value)

  const getCurrentModel = () => AI_MODELS.find((model) => model.id === selectedModel) || AI_MODELS[0]

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMsg = { id: Date.now().toString(), role: "user", content: input }
    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setIsLoading(true)

    try {
      const res = await fetch(`${BACKEND_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input, model: selectedModel, context: "" }),
      })
      if (!res.ok) throw new Error("Network error")
      const data = await res.json()
      const replyMsg = {
        id: Date.now().toString() + "-bot",
        role: "assistant",
        content: data.answer || "Bot chưa trả lời!",
      }
      setMessages((prev) => [...prev, replyMsg])
    } catch {
      setMessages((prev) => [...prev, {
        id: Date.now().toString() + "-bot",
        role: "assistant",
        content: "Lỗi kết nối tới server!",
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl h-[90vh] flex flex-col bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm rounded-xl shadow-2xl border border-slate-200/50 dark:border-slate-700/50">
        <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
          <div className="flex items-center gap-3">
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
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-slate-500" />
            <Badge variant="secondary" className="text-xs">
              {getCurrentModel().name}
            </Badge>
          </div>
        </div>

        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full p-6" ref={scrollAreaRef}>
            <div className="space-y-6">
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <Bot className="h-12 w-12 mx-auto text-slate-400 mb-4" />
                  <h3 className="text-lg font-medium text-slate-600 dark:text-slate-300 mb-2">Chào bạn!</h3>
                  <p className="text-slate-500 dark:text-slate-400">Tôi có thể giúp gì cho bạn</p>
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
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        </div>

        <div className="p-6 border-t border-slate-200 dark:border-slate-700 flex-shrink-0 space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300 whitespace-nowrap">Chọn mô hình:</span>
            <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isLoading}>
              <SelectTrigger className="w-full max-w-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {AI_MODELS.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${model.color}`} />
                      <div className="flex flex-col">
                        <span className="font-medium">{model.name}</span>
                        <span className="text-xs text-slate-500">{model.description}</span>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

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

          <p className="text-xs text-slate-500 dark:text-slate-400 text-center">
            Nhấn Enter để gửi tin nhắn • Mô hình hiện tại: <span className="font-medium">{getCurrentModel().name}</span>
          </p>
        </div>
      </div>
    </div>
  )
}