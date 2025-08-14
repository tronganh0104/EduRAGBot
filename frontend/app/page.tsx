"use client"

import type React from "react"

import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import {
  Send,
  Bot,
  User,
  Cpu,
  MessageSquare,
  Sparkles,
  Settings,
  HelpCircle,
  Zap,
  Brain,
  Sun,
  Moon,
  Plus,
  Trash2,
  MessageSquareText,
  BookOpen,
} from "lucide-react"
import { BACKEND_URL, AI_MODELS } from "./config"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"

interface Reference {
  id: string
  content: string
}

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  model?: string
  references?: Reference[]
}

interface ChatSession {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
}

export default function ProfessionalChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState(AI_MODELS[0].id)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)

  const { theme, setTheme } = useTheme()

  // Load sessions from localStorage on initial mount
  useEffect(() => {
    const storedSessions = localStorage.getItem("chatSessions")
    if (storedSessions) {
      const parsedSessions: ChatSession[] = JSON.parse(storedSessions).map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        messages: session.messages.map((msg: any) => ({ ...msg, timestamp: new Date(msg.timestamp) })),
      }))
      setChatSessions(parsedSessions)
      if (parsedSessions.length > 0) {
        // Load the most recent session by default
        const mostRecentSession = parsedSessions.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())[0]
        setCurrentSessionId(mostRecentSession.id)
        setMessages(mostRecentSession.messages)
      } else {
        createNewSession()
      }
    } else {
      createNewSession()
    }
  }, [])

  // Save sessions to localStorage whenever chatSessions changes
  useEffect(() => {
    localStorage.setItem("chatSessions", JSON.stringify(chatSessions))
  }, [chatSessions])

  // Auto-scroll to bottom when messages or loading state changes
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [messages, isLoading])

  const createNewSession = useCallback(() => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: "Cuộc trò chuyện mới",
      messages: [],
      createdAt: new Date(),
    }
    setChatSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(newSession.id)
    setMessages([])
    setError(null)
    setInput("")
  }, [])

  const loadSession = useCallback(
    (sessionId: string) => {
      const sessionToLoad = chatSessions.find((session) => session.id === sessionId)
      if (sessionToLoad) {
        setCurrentSessionId(sessionId)
        setMessages(sessionToLoad.messages)
        setError(null)
        setInput("")
      }
    },
    [chatSessions],
  )

  const deleteSession = useCallback(
    (sessionId: string) => {
      setChatSessions((prev) => prev.filter((session) => session.id !== sessionId))
      if (currentSessionId === sessionId) {
        // If current session is deleted, create a new one
        createNewSession()
      }
    },
    [currentSessionId, createNewSession],
  )

  const updateCurrentSessionMessages = useCallback(
    (newMessages: Message[]) => {
      setChatSessions((prevSessions) =>
        prevSessions.map((session) =>
          session.id === currentSessionId
            ? {
                ...session,
                messages: newMessages,
                title:
                  session.messages.length === 0 && newMessages.length > 0
                    ? newMessages[0].content.substring(0, 30) + "..."
                    : session.title,
                createdAt: new Date(), // Update timestamp for recent activity
              }
            : session,
        ),
      )
    },
    [currentSessionId],
  )

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
    setError(null)
  }

  const getCurrentModel = () => AI_MODELS.find((model) => model.id === selectedModel) || AI_MODELS[0]

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    }

    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    updateCurrentSessionMessages(updatedMessages) // Update session immediately
    setInput("")
    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch(`${BACKEND_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMsg.content, // Use the content from the userMsg
          model: selectedModel,
          context: "",
        }),
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`)
      }

      const data = await res.json()
      const replyMsg: Message = {
        id: Date.now().toString() + "-bot",
        role: "assistant",
        content: data.answer || "Xin lỗi, tôi không thể trả lời câu hỏi này.",
        timestamp: new Date(),
        model: selectedModel,
        references: data.references || [], // Add references from backend response
      }

      const finalMessages = [...updatedMessages, replyMsg]
      setMessages(finalMessages)
      updateCurrentSessionMessages(finalMessages) // Update session with AI reply
    } catch (err) {
      const errorMsg: Message = {
        id: Date.now().toString() + "-error",
        role: "assistant",
        content: "Xin lỗi, đã có lỗi xảy ra khi kết nối đến server. Vui lòng thử lại sau.",
        timestamp: new Date(),
      }
      const errorMessages = [...updatedMessages, errorMsg]
      setMessages(errorMessages)
      updateCurrentSessionMessages(errorMessages) // Update session with error message
      setError("Không thể kết nối đến server")
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  const ReferenceCard = ({ reference, index }: { reference: Reference; index: number }) => (
    <div className="bg-slate-50 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600 rounded-lg p-4">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
            <span className="text-xs font-medium text-blue-600 dark:text-blue-400">{index + 1}</span>
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">{reference.content}</p>
        </div>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="border-b border-slate-200/60 dark:border-slate-800/60 bg-white/80 dark:bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white dark:border-slate-950"></div>
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-slate-900 to-slate-600 dark:from-slate-100 dark:to-slate-400 bg-clip-text text-transparent">
                    UET AI Assistant
                  </h1>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Trợ lý thông minh về quy chế đào tạo</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <Badge variant="secondary" className="hidden sm:flex items-center space-x-1">
                <Zap className="w-3 h-3" />
                <span className="text-xs font-medium">Online</span>
              </Badge>
              <Button variant="ghost" size="sm" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
                {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6 lg:max-h-[calc(100vh-10rem)] lg:overflow-y-auto">
            <Card className="bg-white/60 dark:bg-slate-900/60 backdrop-blur-sm border-slate-200/60 dark:border-slate-800/60">
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Settings className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                    <h3 className="font-semibold text-slate-900 dark:text-slate-100">Cài đặt mô hình</h3>
                  </div>

                  <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Chọn mô hình AI:</label>
                    <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isLoading}>
                      <SelectTrigger className="w-full bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {AI_MODELS.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            <div className="flex items-center space-x-3 py-1">
                              <div className={`w-3 h-3 rounded-full ${model.color}`} />
                              <div className="flex flex-col justify-center">
                                <span className="font-medium text-sm">{model.name}</span>
                              </div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Mô hình hiện tại:</span>
                      <Badge variant="outline" className="text-xs">
                        {getCurrentModel().name}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Tin nhắn:</span>
                      <span className="font-medium text-slate-900 dark:text-slate-100">{messages.length}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            {/* Chat History Card */}
            <Card className="bg-white/60 dark:bg-slate-900/60 backdrop-blur-sm border-slate-200/60 dark:border-slate-800/60">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <MessageSquareText className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                    <h3 className="font-semibold text-slate-900 dark:text-slate-100">Lịch sử trò chuyện</h3>
                  </div>
                  <Button variant="ghost" size="sm" onClick={createNewSession}>
                    <Plus className="w-4 h-4 mr-1" />
                    <span className="text-xs">Chat mới</span>
                  </Button>
                </div>
                <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
                  {chatSessions.length === 0 && (
                    <p className="text-sm text-slate-500 dark:text-slate-400">Chưa có cuộc trò chuyện nào.</p>
                  )}
                  {chatSessions.map((session) => (
                    <div
                      key={session.id}
                      className={cn(
                        "flex items-center justify-between p-2 rounded-md cursor-pointer transition-colors duration-200",
                        currentSessionId === session.id
                          ? "bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200"
                          : "hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300",
                      )}
                      onClick={() => loadSession(session.id)}
                    >
                      <span className="text-sm font-medium truncate">{session.title}</span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="w-6 h-6 text-slate-400 hover:text-red-500 dark:hover:text-red-400"
                        onClick={(e) => {
                          e.stopPropagation() // Prevent loading session when deleting
                          deleteSession(session.id)
                        }}
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
          {/* Chat Area */}
          <div className="lg:col-span-3 flex flex-col h-[calc(100vh-10rem)]">
            <Card className="h-full bg-white/60 dark:bg-slate-900/60 backdrop-blur-sm border-slate-200/60 dark:border-slate-800/60 flex flex-col">
              {/* Chat Header */}
              <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700 flex-shrink-0">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <MessageSquare className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h2 className="font-semibold text-slate-900 dark:text-slate-100">
                      {currentSessionId
                        ? chatSessions.find((s) => s.id === currentSessionId)?.title || "Cuộc trò chuyện"
                        : "Cuộc trò chuyện"}
                    </h2>
                    <p className="text-xs text-slate-500 dark:text-slate-400">{messages.length} tin nhắn</p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={createNewSession}>
                  <Plus className="w-4 h-4 mr-1" />
                  <span className="text-xs">Chat mới</span>
                </Button>
              </div>

              {/* Fixed Height Chat Messages Container */}
              <div className="flex-1 flex flex-col min-h-0">
                <div className="flex-1 overflow-hidden">
                  <div ref={chatContainerRef} className="h-full overflow-y-auto p-6 space-y-6 scroll-smooth">
                    {messages.length === 0 && (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-center max-w-md mx-auto">
                          <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                            <Sparkles className="w-8 h-8 text-white" />
                          </div>
                          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                            Chào mừng đến với UET AI
                          </h3>
                          <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed mb-4">
                            Tôi là trợ lý AI chuyên về quy chế đào tạo. Hãy đặt câu hỏi để tôi có thể hỗ trợ bạn tốt
                            nhất.
                          </p>
                          <div className="flex flex-wrap gap-2 justify-center">
                            <Badge variant="outline" className="text-xs">
                              <HelpCircle className="w-3 h-3 mr-1" />
                              Quy định điểm số
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              <Settings className="w-3 h-3 mr-1" />
                              Quy trình tốt nghiệp
                            </Badge>
                          </div>
                        </div>
                      </div>
                    )}

                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex gap-4 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        {message.role === "assistant" && (
                          <Avatar className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 flex-shrink-0">
                            <AvatarFallback className="text-white">
                              <Bot className="w-5 h-5" />
                            </AvatarFallback>
                          </Avatar>
                        )}

                        <div className={`max-w-[85%] ${message.role === "user" ? "order-1" : ""}`}>
                          <div
                            className={`rounded-2xl px-5 py-4 shadow-sm ${
                              message.role === "user"
                                ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white"
                                : "bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-700"
                            }`}
                          >
                            <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
                            <div
                              className={`text-xs mt-2 ${
                                message.role === "user" ? "text-blue-100" : "text-slate-500 dark:text-slate-400"
                              }`}
                            >
                              {message.timestamp.toLocaleTimeString("vi-VN", {
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                              {message.model && (
                                <span className="ml-2">• {AI_MODELS.find((m) => m.id === message.model)?.name}</span>
                              )}
                            </div>
                          </div>

                          {/* References Section - Only for assistant messages */}
                          {message.role === "assistant" && message.references && message.references.length > 0 && (
                            <div className="mt-4 space-y-3">
                              <div className="flex items-center space-x-2">
                                <BookOpen className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                                <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
                                  Tài liệu tham khảo ({message.references.length})
                                </h4>
                              </div>
                              <div className="space-y-3">
                                {message.references.map((reference, index) => (
                                  <ReferenceCard key={reference.id} reference={reference} index={index} />
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {message.role === "user" && (
                          <Avatar className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-teal-600 flex-shrink-0 order-2">
                            <AvatarFallback className="text-white">
                              <User className="w-5 h-5" />
                            </AvatarFallback>
                          </Avatar>
                        )}
                      </div>
                    ))}

                    {isLoading && (
                      <div className="flex gap-4 justify-start">
                        <Avatar className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 flex-shrink-0">
                          <AvatarFallback className="text-white">
                            <Bot className="w-5 h-5" />
                          </AvatarFallback>
                        </Avatar>
                        <div className="bg-white dark:bg-slate-800 rounded-2xl px-5 py-4 border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-2">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                              <div
                                className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                                style={{ animationDelay: "0.1s" }}
                              ></div>
                              <div
                                className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                                style={{ animationDelay: "0.2s" }}
                              ></div>
                            </div>
                            <span className="text-sm text-slate-600 dark:text-slate-400">AI đang suy nghĩ...</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* This div acts as the scroll target */}
                    <div ref={messagesEndRef} className="h-1" />
                  </div>
                </div>

                {/* Fixed Input Area */}
                <div className="border-t border-slate-200 dark:border-slate-700 p-6 bg-slate-50/50 dark:bg-slate-800/50 flex-shrink-0">
                  {error && (
                    <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                      <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                    </div>
                  )}

                  <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="flex gap-3">
                      <div className="flex-1 relative">
                        <Input
                          value={input}
                          onChange={handleInputChange}
                          onKeyPress={handleKeyPress}
                          placeholder="Nhập câu hỏi của bạn về quy chế đào tạo..."
                          className="pr-12 py-3 text-base bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 focus:border-blue-500 dark:focus:border-blue-400 rounded-xl"
                          disabled={isLoading}
                        />
                        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                          <Cpu className="w-4 h-4 text-slate-400" />
                        </div>
                      </div>
                      <Button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
                      >
                        <Send className="w-5 h-5" />
                      </Button>
                    </div>

                    <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                      <span>Nhấn Enter để gửi • Shift + Enter để xuống dòng</span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${getCurrentModel().color}`}></div>
                        <span>{getCurrentModel().name}</span>
                      </div>
                    </div>
                  </form>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
