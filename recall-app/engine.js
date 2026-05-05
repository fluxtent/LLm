class RecallEngine {
  constructor(memory) {
    this.memory = memory;
    this.abortController = null;
    this.isGenerating = false;
  }

  apiUrl(path) {
    return window.MedBriefRuntime
      ? window.MedBriefRuntime.apiUrl(path)
      : (path.startsWith('/') ? path : `/${path}`);
  }

  buildHeaders() {
    const headers = {
      'Content-Type': 'application/json'
    };

    if (RECALL_CONFIG.API_KEY) {
      headers.Authorization = `Bearer ${RECALL_CONFIG.API_KEY}`;
    }

    return headers;
  }

  formatErrorDetail(detail) {
    if (!detail) return '';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map(item => {
          if (typeof item === 'string') return item;
          if (item?.msg) return item.msg;
          try {
            return JSON.stringify(item);
          } catch {
            return String(item);
          }
        })
        .filter(Boolean);
      return messages.join(' | ');
    }
    if (typeof detail === 'object') {
      if (detail.msg) return detail.msg;
      try {
        return JSON.stringify(detail);
      } catch {
        return String(detail);
      }
    }
    return String(detail);
  }

  detectCrisis(text) {
    const lower = (text || '').toLowerCase();
    const highConfidence = RECALL_CONFIG.CRISIS_HIGH_CONFIDENCE_KEYWORDS || RECALL_CONFIG.CRISIS_KEYWORDS || [];
    if (highConfidence.some(keyword => lower.includes(keyword))) {
      return true;
    }

    const keywordMatches = new Set((RECALL_CONFIG.CRISIS_KEYWORDS || []).filter(keyword => lower.includes(keyword)));
    const distressMatches = new Set((RECALL_CONFIG.CRISIS_DISTRESS_SIGNALS || []).filter(signal => lower.includes(signal)));
    return keywordMatches.size >= 2 || (keywordMatches.size >= 1 && distressMatches.size >= 1);
  }

  detectMode(text) {
    const lower = text.toLowerCase();
    
    const psychKeywords = ['anxious', 'depressed', 'feeling', 'overwhelmed', 'struggling', 'therapist', 'panic', 'mental', 'emotional', 'burnout', 'lonely', 'ashamed', 'hurt', 'hurting', 'grief', 'lost', 'failure', 'failed', 'can\'t think', 'cant think', 'all over', 'hopeless', 'worthless', 'ruined'];
    const healthKeywords = ['symptom', 'medication', 'diagnosis', 'doctor', 'pain', 'condition', 'treatment', 'hospital', 'medical', 'insomnia', 'headache', 'health'];
    const portfolioKeywords = ['your project', 'medbrief', 'what is this', 'who made', 'about you', 'website', 'portfolio', 'product', 'feature'];
    
    if (this.detectCrisis(lower)) return 'crisis';
    if (psychKeywords.some(kw => lower.includes(kw))) return 'psych';
    if (healthKeywords.some(kw => lower.includes(kw))) return 'health';
    if (portfolioKeywords.some(kw => lower.includes(kw))) return 'portfolio';
    
    return 'general';
  }

  modeTag(mode) {
    const safeMode = (mode || 'general').toUpperCase();
    return `[MODE:${safeMode}]`;
  }

  cleanResponse(text) {
    text = text.replace(/<eos>/gi, '').replace(/<[^>]+>/gi, '');
    text = text.replace(/\s+/g, ' ').trim();
    
    const lastPunct = Math.max(text.lastIndexOf('.'), text.lastIndexOf('!'), text.lastIndexOf('?'));
    if (lastPunct > text.length * 0.6) {
      text = text.slice(0, lastPunct + 1);
    }
    
    return text;
  }

  async createErrorFromResponse(response) {
    const requestId = response.headers.get('X-Request-ID') || '';
    let detail = `API error: ${response.status}`;

    try {
      const text = await response.text();
      if (text) {
        try {
          const payload = JSON.parse(text);
          detail = this.formatErrorDetail(payload.detail || payload.error || detail) || detail;
        } catch {
          detail = text;
        }
      }
    } catch {
    }

    const error = new Error(detail);
    error.status = response.status;
    error.requestId = requestId;
    return error;
  }

  buildMessages(conversationId) {
    return this.memory.getContextWindow(
      conversationId,
      RECALL_CONFIG.MAX_CONTEXT_MESSAGES
    );
  }

  buildMetadata() {
    return {
      user_id: this.memory.getUserId(),
      user_profile: this.memory.getUserProfile()
    };
  }

  async syncProfile() {
    if (!RECALL_CONFIG.ENABLED_FEATURES?.profileEnabled) return;
    const profile = this.memory.getUserProfile();
    try {
      const response = await fetch(this.apiUrl('/v1/profile'), {
        method: 'POST',
        credentials: 'include',
        headers: this.buildHeaders(),
        body: JSON.stringify({
          user_id: profile.user_id,
          profile
        })
      });
      if (!response.ok) return;
      const payload = await response.json();
      if (payload?.profile) {
        this.memory.setServerProfile(payload.profile);
      }
    } catch {
    }
  }

  async initializeSession(conversationId) {
    if (!RECALL_CONFIG.ENABLED_FEATURES?.profileEnabled || !conversationId) return;
    try {
      const response = await fetch(this.apiUrl('/v1/session/init'), {
        method: 'POST',
        credentials: 'include',
        headers: this.buildHeaders(),
        body: JSON.stringify({
          user_id: this.memory.getUserId(),
          session_id: conversationId
        })
      });
      if (!response.ok) return;
      const payload = await response.json();
      if (payload?.profile) {
        this.memory.setServerProfile(payload.profile);
      }
      if (payload?.memory_summary) {
        this.memory.addSessionSummary(payload.memory_summary);
      }
    } catch {
    }
  }

  async maybeSummarizeConversation(conversationId) {
    if (!this.memory.needsSummarization(conversationId)) return;
    try {
      const conversation = this.memory.getConversation(conversationId);
      if (!conversation) return;
      const response = await fetch(this.apiUrl('/v1/memory/summarize'), {
        method: 'POST',
        credentials: 'include',
        headers: this.buildHeaders(),
        body: JSON.stringify({
          user_id: this.memory.getUserId(),
          session_id: conversationId,
          messages: conversation.messages.map(message => ({
            role: message.role,
            content: message.content
          }))
        })
      });
      if (!response.ok) return;
      const payload = await response.json();
      if (payload?.summary) {
        this.memory.applyServerSummary(conversationId, payload.summary);
      }
    } catch {
    }
  }

  buildRequestMessages(conversationId, userMessage, mode, isCrisis) {
    const contextMessages = this.buildMessages(conversationId).map(message => ({ ...message }));
    const taggedUserMessage = `${this.modeTag(mode)} ${userMessage}`.trim();

    for (let i = contextMessages.length - 1; i >= 0; i--) {
      if (contextMessages[i].role === 'user') {
        contextMessages[i].content = taggedUserMessage;
        break;
      }
    }

    const requestMessages = [...contextMessages];

    if (isCrisis) {
      requestMessages.push({
        role: 'system',
        content: 'SAFETY ALERT: The user may be expressing crisis-level distress. Follow the crisis protocol immediately. Be gentle, direct, and provide emergency resources. Do not continue casual exploration.'
      });
    }

    return requestMessages;
  }

  async sendMessage(conversationId, userMessage, onChunk, onComplete, onError) {
    return this.dispatchMessage({
      conversationId,
      userMessage,
      onChunk,
      onComplete,
      onError,
      persistUserMessage: true
    });
  }

  async retryLastMessage(conversationId, onChunk, onComplete, onError) {
    const userMessage = this.memory.latestUserPrompt(conversationId);
    if (!userMessage) {
      onError(new Error('No previous user message is available to retry.'));
      return;
    }

    return this.dispatchMessage({
      conversationId,
      userMessage,
      onChunk,
      onComplete,
      onError,
      persistUserMessage: false
    });
  }

  async dispatchMessage({ conversationId, userMessage, onChunk, onComplete, onError, persistUserMessage }) {
    if (this.isGenerating) {
      this.cancel();
    }

    this.isGenerating = true;
    this.abortController = new AbortController();
    await this.initializeSession(conversationId);
    await this.syncProfile();

    const isCrisis = this.detectCrisis(userMessage);
    const mode = this.detectMode(userMessage);

    if (persistUserMessage) {
      this.memory.addMessage(conversationId, 'user', userMessage);
    }

    const messages = this.buildRequestMessages(conversationId, userMessage, mode, isCrisis);
    const memorySummary = this.memory.buildMemoryContext();
    const requestId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : `req-${Date.now()}`;

    try {
      if (RECALL_CONFIG.STREAM) {
        await this.streamRequest(messages, conversationId, memorySummary, isCrisis, mode, requestId, userMessage, onChunk, onComplete);
      } else {
        await this.standardRequest(messages, conversationId, memorySummary, isCrisis, mode, requestId, userMessage, onChunk, onComplete);
      }
      await this.maybeSummarizeConversation(conversationId);
    } catch (error) {
      if (error.name === 'AbortError') {
        this.isGenerating = false;
        return;
      }
      this.isGenerating = false;
      console.error('Engine error:', error);
      onError(error);
    }
  }

  async streamRequest(messages, conversationId, memorySummary, isCrisis, mode, requestId, userMessage, onChunk, onComplete) {
    const response = await fetch(RECALL_CONFIG.API_ENDPOINT, {
      method: 'POST',
      credentials: 'include',
      headers: this.buildHeaders(),
      body: JSON.stringify({
        model: RECALL_CONFIG.MODEL,
        messages,
        max_tokens: RECALL_CONFIG.MAX_TOKENS,
        temperature: RECALL_CONFIG.TEMPERATURE,
        stream: true,
        mode: mode,
        conversation_id: conversationId,
        memory_summary: memorySummary || undefined,
        request_id: requestId,
        metadata: this.buildMetadata()
      }),
      signal: this.abortController.signal
    });

    if (!response.ok) {
      throw await this.createErrorFromResponse(response);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = '';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;
        if (!trimmed.startsWith('data: ')) continue;

        try {
          const json = JSON.parse(trimmed.slice(6));
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) {
            fullContent += delta;
            onChunk(delta);
          }
        } catch {
          continue;
        }
      }
    }

    const finalContent = this.cleanResponse(fullContent);
    if (!finalContent) {
      throw new Error('The backend returned an empty response.');
    }

    this.memory.addMessage(conversationId, 'assistant', finalContent);
    this.isGenerating = false;
    onComplete(finalContent);
  }

  async standardRequest(messages, conversationId, memorySummary, isCrisis, mode, requestId, userMessage, onChunk, onComplete) {
    const response = await fetch(RECALL_CONFIG.API_ENDPOINT, {
      method: 'POST',
      credentials: 'include',
      headers: this.buildHeaders(),
      body: JSON.stringify({
        model: RECALL_CONFIG.MODEL,
        messages,
        max_tokens: RECALL_CONFIG.MAX_TOKENS,
        temperature: RECALL_CONFIG.TEMPERATURE,
        stream: false,
        mode: mode,
        conversation_id: conversationId,
        memory_summary: memorySummary || undefined,
        request_id: requestId,
        metadata: this.buildMetadata()
      }),
      signal: this.abortController.signal
    });

    if (!response.ok) {
      throw await this.createErrorFromResponse(response);
    }

    const data = await response.json();
    const content = this.cleanResponse(data.choices?.[0]?.message?.content || '');
    if (!content) {
      throw new Error('The backend returned an empty response.');
    }

    this.typeResponse(content, conversationId, isCrisis, onChunk, onComplete);
  }

  typeResponse(content, conversationId, isCrisis, onChunk, onComplete) {
    let i = 0;
    const chunkSize = 3;
    const interval = 12;

    const typeNext = () => {
      if (!this.isGenerating) return;

      if (i < content.length) {
        const chunk = content.substring(i, Math.min(i + chunkSize, content.length));
        onChunk(chunk);
        i += chunkSize;
        setTimeout(typeNext, interval);
      } else {
        this.memory.addMessage(conversationId, 'assistant', content);
        this.isGenerating = false;
        onComplete(content);
      }
    };

    typeNext();
  }

  cancel() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.isGenerating = false;
  }

  parseMarkdown(text) {
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');

    html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    html = html.replace(/^---$/gm, '<hr>');

    const paragraphs = html.split(/\n\n+/).filter(p => p.trim());
    html = paragraphs.map(p => {
      p = p.trim();
      if (p.startsWith('<ul>') || p.startsWith('<hr>') || p.startsWith('<ol>')) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
  }
}
