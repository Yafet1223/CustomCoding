import { useMemo, useRef, useState } from 'react'
import './App.css'

const starterPrompts = [
  'Create a Python CLI calculator in the sandbox',
  'Read the project files and suggest improvements',
  'Add tests for an existing file',
]

const formatJson = (value) => {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function Message({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar" aria-hidden="true">{message.role === 'assistant' ? 'AI' : 'Y'}</div>
      <div className="bubble">
        <p>{message.content}</p>
      </div>
    </article>
  )
}

function ApprovalCard({ pending, onDecision, busy }) {
  if (!pending) return null

  return (
    <section className="approvalCard" aria-live="polite">
      <div>
        <span className="eyebrow">Approval required</span>
        <h2>{pending.action}</h2>
      </div>
      <pre>{formatJson(pending.details)}</pre>
      <div className="approvalActions">
        <button className="secondaryButton" onClick={() => onDecision('reject')} disabled={busy}>
          Reject
        </button>
        <button onClick={() => onDecision('approve')} disabled={busy}>
          Approve
        </button>
      </div>
    </section>
  )
}

function App() {
  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: 'assistant',
      content:
        'Tell me what you want to build or change. I can inspect files, propose edits, and pause for approval before writing or running commands.',
    },
  ])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState('')
  const [pending, setPending] = useState(null)
  const [isBusy, setIsBusy] = useState(false)
  const [error, setError] = useState('')
  const textareaRef = useRef(null)

  const canSend = useMemo(() => input.trim().length > 0 && !isBusy && !pending, [input, isBusy, pending])

  const addAssistantResponse = (data) => {
    setThreadId(data.thread_id)
    setPending(data.status === 'paused' ? data.pending : null)

    const content =
  data.status === 'paused'
    ? 'I need your approval before continuing with this action.'
    : data.response?.trim() || 'No response came back from the agent.'

    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'assistant',
        content,
        trace: data.trace || [],
      },
    ])
  }

  const callApi = async (url, body) => {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data.detail || 'The backend returned an error.')
    }
    return data
  }

  const sendMessage = async (overrideText) => {
    const text = (overrideText || input).trim()
    if (!text || isBusy || pending) return

    setInput('')
    setError('')
    setIsBusy(true)
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'user', content: text }])

    try {
      const data = await callApi('/api/chat', {
        message: text,
        thread_id: threadId || undefined,
      })
      addAssistantResponse(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsBusy(false)
      textareaRef.current?.focus()
    }
  }

  const handleDecision = async (decision) => {
    if (!pending || !threadId || isBusy) return

    setError('')
    setIsBusy(true)
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: 'user', content: `${decision === 'approve' ? 'Approved' : 'Rejected'}: ${pending.action}` },
    ])

    try {
      const data = await callApi('/api/resume', { thread_id: threadId, decision })
      addAssistantResponse(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsBusy(false)
      textareaRef.current?.focus()
    }
  }

  const resetChat = () => {
    setMessages([
      {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Fresh thread started. What should we work on?',
      },
    ])
    setInput('')
    setThreadId('')
    setPending(null)
    setError('')
  }

  return (
    <main className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brandMark">CC</span>
          <div>
            <h1>Custom Coding</h1>
            <p>AI workspace</p>
          </div>
        </div>
        <button className="newChatButton" onClick={resetChat}>New chat</button>
        <div className="sidebarSection">
          <span className="eyebrow">Try asking</span>
          {starterPrompts.map((prompt) => (
            <button className="promptButton" key={prompt} onClick={() => sendMessage(prompt)} disabled={isBusy || Boolean(pending)}>
              {prompt}
            </button>
          ))}
        </div>
        <div className="statusPanel">
          <span className={`statusDot ${pending ? 'paused' : 'ready'}`} />
          <span>{pending ? 'Waiting for approval' : isBusy ? 'Thinking' : 'Ready'}</span>
        </div>
      </aside>

      <section className="chatPanel">
        <header className="chatHeader">
          <div>
            <span className="eyebrow">Conversation</span>
            <h2>Coding Assistant</h2>
          </div>
          {threadId && <code title={threadId}>Thread {threadId.slice(0, 8)}</code>}
        </header>

        <div className="messageList">
          {messages.map((message) => (
            <Message message={message} key={message.id} />
          ))}
          {isBusy && (
            <article className="message assistant">
              <div className="avatar">AI</div>
              <div className="bubble typing">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}
        </div>

        <div className="composerArea">
          <ApprovalCard pending={pending} onDecision={handleDecision} busy={isBusy} />
          {error && <div className="errorBanner">{error}</div>}
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault()
              sendMessage()
            }}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  sendMessage()
                }
              }}
              placeholder={pending ? 'Approve or reject the pending action to continue' : 'Message the coding agent...'}
              disabled={isBusy || Boolean(pending)}
              rows={1}
            />
            <button type="submit" disabled={!canSend} aria-label="Send message">Send</button>
          </form>
        </div>
      </section>
    </main>
  )
}

export default App
