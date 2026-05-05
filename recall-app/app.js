document.addEventListener('DOMContentLoaded', () => {
  const memory = new RecallMemory();
  const engine = new RecallEngine(memory);
  const apiUrl = path => window.MedBriefRuntime
    ? window.MedBriefRuntime.apiUrl(path)
    : (path.startsWith('/') ? path : `/${path}`);

  let currentConversationId = null;
  let currentTheme = localStorage.getItem('recall_theme') || 'dark';
  let runtimeConfig = null;
  let currentApiKeyId = localStorage.getItem('recall_api_key_id') || '';

  const $ = id => document.getElementById(id);
  const welcomeScreen = $('welcome-screen');
  const chatMessages = $('chat-messages');
  const messagesContainer = $('messages-container');
  const messageInput = $('message-input');
  const sendBtn = $('send-btn');
  const newChatBtn = $('new-chat-btn');
  const sidebarToggle = $('sidebar-toggle');
  const sidebar = $('sidebar');
  const conversationList = $('conversation-list');
  const themeToggle = $('theme-toggle');
  const themeIconDark = $('theme-icon-dark');
  const themeIconLight = $('theme-icon-light');
  const moodCheckBtn = $('mood-check-btn');
  const moodModal = $('mood-modal');
  const moodClose = $('mood-close');
  const crisisBtn = $('crisis-btn');
  const crisisModal = $('crisis-modal');
  const crisisClose = $('crisis-close');
  const headerStatus = $('header-status');
  const welcomePrompts = $('welcome-prompts');
  const settingsBtn = $('settings-btn');
  const settingsModal = $('settings-modal');
  const settingsClose = $('settings-close');
  const generateApiKeyBtn = $('generate-api-key');
  const revokeApiKeyBtn = $('revoke-api-key');
  const apiKeyDisplay = $('api-key-display');
  const apiKeyValue = $('api-key-value');
  const apiKeyCopy = $('api-key-copy');
  const developerAccessSection = $('developer-access-section');
  const clearMemoryBtn = $('clear-memory');
  const prefMaxTokens = $('pref-max-tokens');
  const prefTemperature = $('pref-temperature');
  const prefTerminology = $('pref-terminology');
  const prefResponseLength = $('pref-response-length');
  const prefTone = $('pref-tone');

  async function init() {
    applyTheme(currentTheme);
    renderConversationList();
    setupEventListeners();
    autoResizeInput();
    loadPreferences();
    await loadRuntimeConfig();
    await engine.syncProfile();
    loadSavedApiKey();

    if (window.innerWidth <= 768) {
      sidebar.classList.add('collapsed');
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    currentTheme = theme;
    localStorage.setItem('recall_theme', theme);

    if (theme === 'dark') {
      themeIconDark.style.display = '';
      themeIconLight.style.display = 'none';
    } else {
      themeIconDark.style.display = 'none';
      themeIconLight.style.display = '';
    }
  }

  function setupEventListeners() {
    sendBtn.addEventListener('click', handleSend);

    messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    messageInput.addEventListener('input', () => {
      autoResizeInput();
      sendBtn.disabled = !messageInput.value.trim();
    });

    newChatBtn.addEventListener('click', startNewConversation);

    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
    });

    themeToggle.addEventListener('click', () => {
      applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });

    moodCheckBtn.addEventListener('click', () => {
      moodModal.style.display = 'flex';
    });
    moodClose.addEventListener('click', () => {
      moodModal.style.display = 'none';
    });
    moodModal.addEventListener('click', (e) => {
      if (e.target === moodModal) moodModal.style.display = 'none';
    });

    document.querySelectorAll('.mood-option').forEach(btn => {
      btn.addEventListener('click', () => handleMoodSelect(btn));
    });

    crisisBtn.addEventListener('click', () => {
      crisisModal.style.display = 'flex';
    });
    crisisClose.addEventListener('click', () => {
      crisisModal.style.display = 'none';
    });
    crisisModal.addEventListener('click', (e) => {
      if (e.target === crisisModal) crisisModal.style.display = 'none';
    });

    settingsBtn.addEventListener('click', () => {
      populateSettingsPanel();
      settingsModal.style.display = 'flex';
    });
    settingsClose.addEventListener('click', () => {
      settingsModal.style.display = 'none';
    });
    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) settingsModal.style.display = 'none';
    });

    if (generateApiKeyBtn) {
      generateApiKeyBtn.addEventListener('click', handleGenerateApiKey);
    }
    if (revokeApiKeyBtn) {
      revokeApiKeyBtn.addEventListener('click', handleRevokeApiKey);
    }
    if (apiKeyCopy) {
      apiKeyCopy.addEventListener('click', handleCopyApiKey);
    }
    clearMemoryBtn.addEventListener('click', handleClearMemory);

    prefMaxTokens.addEventListener('change', () => {
      const val = Math.max(50, Math.min(2000, parseInt(prefMaxTokens.value) || 150));
      prefMaxTokens.value = val;
      RECALL_CONFIG.MAX_TOKENS = val;
      localStorage.setItem('recall_max_tokens', val);
    });

    prefTemperature.addEventListener('change', () => {
      const val = Math.max(0, Math.min(2, parseFloat(prefTemperature.value) || 0.8));
      prefTemperature.value = val;
      RECALL_CONFIG.TEMPERATURE = val;
      localStorage.setItem('recall_temperature', val);
    });

    prefTerminology.addEventListener('change', async () => {
      memory.updatePreferences({ terminology: prefTerminology.value });
      await engine.syncProfile();
    });

    prefResponseLength.addEventListener('change', async () => {
      memory.updatePreferences({ response_length: prefResponseLength.value });
      await engine.syncProfile();
    });

    prefTone.addEventListener('change', async () => {
      memory.updatePreferences({ tone: prefTone.value });
      await engine.syncProfile();
    });

    welcomePrompts.addEventListener('click', (e) => {
      const card = e.target.closest('.prompt-card');
      if (card) {
        const prompt = card.dataset.prompt;
        messageInput.value = prompt;
        autoResizeInput();
        sendBtn.disabled = false;
        handleSend();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        moodModal.style.display = 'none';
        crisisModal.style.display = 'none';
        settingsModal.style.display = 'none';
      }
    });
  }

  function autoResizeInput() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + 'px';
  }

  function clearRequestErrors() {
    document.querySelectorAll('.request-error').forEach(el => el.remove());
  }

  function discardAssistantDraft() {
    if (currentAssistantEl) {
      currentAssistantEl.remove();
      currentAssistantEl = null;
      currentAssistantContent = '';
    }
  }

  function describeRequestError(error) {
    const base = (error?.message || '').trim() || 'Something went wrong - please try again.';
    const requestId = error?.requestId ? ` Request ID: ${error.requestId}.` : '';
    return `${base}${requestId}`;
  }

  function appendRequestError(error) {
    const el = document.createElement('div');
    el.className = 'request-error';
    el.innerHTML = `
      <div class="request-error-copy">
        <strong>Something went wrong.</strong>
        <span>${escapeHtml(describeRequestError(error))}</span>
      </div>
    `;

    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const retryBtn = document.createElement('button');
    retryBtn.className = 'feedback-btn retry-btn';
    retryBtn.textContent = 'Retry last message';
    retryBtn.addEventListener('click', async () => {
      retryBtn.disabled = true;
      el.remove();
      await requestAssistantRetry();
    });

    actions.appendChild(retryBtn);
    el.appendChild(actions);
    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  async function requestAssistantReply(text) {
    clearRequestErrors();
    setStatus('thinking');
    showTypingIndicator();

    await engine.sendMessage(
      currentConversationId,
      text,
      (chunk) => {
        hideTypingIndicator();
        appendAssistantChunk(chunk);
        scrollToBottom();
      },
      (fullContent) => {
        finalizeAssistantMessage(fullContent);
        setStatus('ready');
        renderConversationList();
        scrollToBottom();
      },
      (error) => {
        hideTypingIndicator();
        console.error(error);
        if (currentAssistantContent.trim()) {
          finalizeAssistantMessage(currentAssistantContent);
        } else {
          discardAssistantDraft();
        }
        setStatus('degraded', 'Something went wrong - retry is available');
        renderConversationList();
        appendRequestError(error);
      }
    );
  }

  async function requestAssistantRetry() {
    clearRequestErrors();
    setStatus('thinking');
    showTypingIndicator();

    await engine.retryLastMessage(
      currentConversationId,
      (chunk) => {
        hideTypingIndicator();
        appendAssistantChunk(chunk);
        scrollToBottom();
      },
      (fullContent) => {
        finalizeAssistantMessage(fullContent);
        setStatus('ready');
        renderConversationList();
        scrollToBottom();
      },
      (error) => {
        hideTypingIndicator();
        console.error(error);
        if (currentAssistantContent.trim()) {
          finalizeAssistantMessage(currentAssistantContent);
        } else {
          discardAssistantDraft();
        }
        setStatus('degraded', 'Something went wrong - retry is available');
        renderConversationList();
        appendRequestError(error);
      }
    );
  }

  async function handleSend() {
    const text = messageInput.value.trim();
    if (!text || engine.isGenerating) return;

    if (!currentConversationId) {
      const conv = memory.createConversation();
      currentConversationId = conv.id;
      renderConversationList();
    }

    showChatView();
    appendUserMessage(text);

    messageInput.value = '';
    autoResizeInput();
    sendBtn.disabled = true;

    await requestAssistantReply(text);
  }

  function startNewConversation() {
    currentConversationId = null;
    showWelcomeView();
    renderConversationList();
    messageInput.value = '';
    autoResizeInput();
    sendBtn.disabled = true;
    messageInput.focus();
  }

  function showChatView() {
    welcomeScreen.style.display = 'none';
    chatMessages.style.display = 'flex';
    chatMessages.style.flexDirection = 'column';
    chatMessages.style.flex = '1';
  }

  function showWelcomeView() {
    welcomeScreen.style.display = 'flex';
    chatMessages.style.display = 'none';
    messagesContainer.innerHTML = '';
  }

  function appendUserMessage(text) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgEl = document.createElement('div');
    msgEl.className = 'message user';
    msgEl.innerHTML = `
      <div class="message-avatar">You</div>
      <div>
        <div class="message-content"><p>${escapeHtml(text)}</p></div>
        <div class="message-meta"><span class="message-time">${timeStr}</span></div>
      </div>
    `;
    messagesContainer.appendChild(msgEl);
    scrollToBottom();
  }

  let currentAssistantEl = null;
  let currentAssistantContent = '';

  function appendAssistantChunk(chunk) {
    if (!currentAssistantEl) {
      currentAssistantContent = '';
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      currentAssistantEl = document.createElement('div');
      currentAssistantEl.className = 'message assistant';
      currentAssistantEl.innerHTML = `
        <div class="message-avatar">
          <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
            <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
            <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="message-content"></div>
          <div class="message-meta"><span class="message-time">${timeStr}</span></div>
        </div>
      `;
      messagesContainer.appendChild(currentAssistantEl);
    }

    currentAssistantContent += chunk;
    const contentEl = currentAssistantEl.querySelector('.message-content');
    contentEl.innerHTML = engine.parseMarkdown(currentAssistantContent);
  }

  function finalizeAssistantMessage(fullContent) {
    if (currentAssistantEl) {
      const contentEl = currentAssistantEl.querySelector('.message-content');
      contentEl.innerHTML = engine.parseMarkdown(fullContent);
      const conversationId = currentConversationId;
      if (RECALL_CONFIG.ENABLED_FEATURES?.feedbackEnabled && conversationId) {
        addFeedbackControls(currentAssistantEl, conversationId, fullContent);
      }

      if (engine.detectCrisis(fullContent) || containsCrisisResources(fullContent)) {
        addSafetyBanner(currentAssistantEl);
      }
    }
    currentAssistantEl = null;
    currentAssistantContent = '';
  }

  function containsCrisisResources(text) {
    const lower = (text || '').toLowerCase();
    return lower.includes('988') || lower.includes('741741') || lower.includes('emergency services');
  }

  function addSafetyBanner(messageEl) {
    const banner = document.createElement('div');
    banner.className = 'safety-banner';
    banner.innerHTML = `<p>💛 MedBrief AI cares about your safety. If you need immediate help, please call <a href="tel:988"><strong>988</strong></a> or <a href="tel:911"><strong>911</strong></a>.</p>`;
    messageEl.querySelector('.message-content').appendChild(banner);
  }

  function showTypingIndicator() {
    hideTypingIndicator();
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
      <div class="message-avatar">
        <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
          <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
          <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="typing-bubble">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
    messagesContainer.appendChild(indicator);
    scrollToBottom();
  }

  function hideTypingIndicator() {
    const existing = document.getElementById('typing-indicator');
    if (existing) existing.remove();
  }

  function appendSystemMessage(text) {
    const el = document.createElement('div');
    el.className = 'session-divider';
    el.innerHTML = `<span>${escapeHtml(text)}</span>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  function setStatus(status, detail = '') {
    const dot = headerStatus.querySelector('.status-dot');
    const span = headerStatus.querySelector('span');

    if (status === 'thinking') {
      dot.style.background = 'var(--accent-primary)';
      span.textContent = 'MedBrief AI is thinking…';
    } else if (status === 'degraded') {
      dot.style.background = '#F59E0B';
      span.textContent = detail || 'MedBrief AI is in degraded mode';
    } else {
      dot.style.background = 'var(--success)';
      span.textContent = 'MedBrief AI is ready';
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function renderConversationList() {
    const convs = memory.getAllConversations();

    conversationList.innerHTML = '';

    const sLabel = document.createElement('div');
    sLabel.className = 'sidebar-section-label';
    sLabel.textContent = 'Recent';
    conversationList.appendChild(sLabel);

    if (convs.length === 0) {
      return;
    }

    convs.forEach(conv => {
      const item = document.createElement('div');
      item.className = 'conversation-item' + (conv.id === currentConversationId ? ' active' : '');
      item.innerHTML = `
        <span class="conv-title">${escapeHtml(conv.title)}</span>
        <button class="conv-delete" data-id="${conv.id}" title="Delete conversation">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2.5 2.5L9.5 9.5M9.5 2.5L2.5 9.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
          </svg>
        </button>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.closest('.conv-delete')) return;
        loadConversation(conv.id);
      });

      item.querySelector('.conv-delete').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteConversation(conv.id);
      });

      conversationList.appendChild(item);
    });
  }

  function loadConversation(id) {
    const conv = memory.getConversation(id);
    if (!conv) return;

    currentConversationId = id;
    showChatView();
    messagesContainer.innerHTML = '';

    conv.messages.forEach(msg => {
      if (msg.role === 'user') {
        appendExistingUserMessage(msg);
      } else if (msg.role === 'assistant') {
        appendExistingAssistantMessage(msg);
      }
    });

    renderConversationList();
    scrollToBottom();

    if (window.innerWidth <= 768) {
      sidebar.classList.add('collapsed');
    }
  }

  function appendExistingUserMessage(msg) {
    const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const el = document.createElement('div');
    el.className = 'message user';
    el.innerHTML = `
      <div class="message-avatar">You</div>
      <div>
        <div class="message-content"><p>${escapeHtml(msg.content)}</p></div>
        <div class="message-meta"><span class="message-time">${time}</span></div>
      </div>
    `;
    messagesContainer.appendChild(el);
  }

  function appendExistingAssistantMessage(msg) {
    const time = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
      <div class="message-avatar">
        <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
          <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
          <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="message-content">${engine.parseMarkdown(msg.content)}</div>
        <div class="message-meta"><span class="message-time">${time}</span></div>
      </div>
    `;
    if (RECALL_CONFIG.ENABLED_FEATURES?.feedbackEnabled && currentConversationId) {
      addFeedbackControls(el, currentConversationId, msg.content);
    }
    if (engine.detectCrisis(msg.content) || containsCrisisResources(msg.content)) {
      addSafetyBanner(el);
    }
    messagesContainer.appendChild(el);
  }

  function addFeedbackControls(messageEl, conversationId, assistantText) {
    if (messageEl.querySelector('.message-actions')) return;

    const actions = document.createElement('div');
    actions.className = 'message-actions';

    const upBtn = document.createElement('button');
    upBtn.className = 'feedback-btn';
    upBtn.innerHTML = '👍 Helpful';

    const downBtn = document.createElement('button');
    downBtn.className = 'feedback-btn';
    downBtn.innerHTML = '👎 Not quite';

    const submit = async (rating, button) => {
      const lastUserPrompt = memory.latestUserPrompt(conversationId);
      upBtn.classList.remove('active');
      downBtn.classList.remove('active');
      button.classList.add('active');

      try {
        await fetch(apiUrl('/v1/feedback'), {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: memory.getUserId(),
            conversation_id: conversationId,
            rating,
            prompt: lastUserPrompt,
            response: assistantText,
            mode: engine.detectMode(lastUserPrompt)
          })
        });
      } catch {
      }
    };

    upBtn.addEventListener('click', () => submit('up', upBtn));
    downBtn.addEventListener('click', () => submit('down', downBtn));

    actions.appendChild(upBtn);
    actions.appendChild(downBtn);
    messageEl.querySelector('.message-content').appendChild(actions);
  }

  function deleteConversation(id) {
    memory.deleteConversation(id);
    if (currentConversationId === id) {
      startNewConversation();
    }
    renderConversationList();
  }

  function handleMoodSelect(btn) {
    document.querySelectorAll('.mood-option').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');

    const value = parseInt(btn.dataset.value);
    memory.recordMood(value);

    setTimeout(() => {
      moodModal.style.display = 'none';
      btn.classList.remove('selected');

      const moodLabel = RECALL_CONFIG.MOOD_LABELS[value];
      const moodResponses = {
        5: "It's wonderful to hear you're feeling good. What's contributing to that today?",
        4: "Glad you're holding steady. Anything specific making today feel okay?",
        3: "A 'meh' day — those are valid too. Is there anything in particular keeping you in the middle?",
        2: "I'm sorry you're feeling low. Would it help to talk about what's pulling you down?",
        1: "I hear you. It takes courage to acknowledge when things are hard. I'm here. What feels heaviest right now?"
      };

      if (!currentConversationId) {
        const conv = memory.createConversation();
        currentConversationId = conv.id;
        memory.updateConversation(conv.id, { title: `Mood check-in: ${moodLabel}` });
      }

      showChatView();

      const systemMsg = `[User completed a mood check-in and selected: "${moodLabel}" (${value}/5)]`;
      memory.addMessage(currentConversationId, 'user', `I just did a mood check-in. I'm feeling ${moodLabel} right now.`);
      appendUserMessage(`I just did a mood check-in. I'm feeling ${moodLabel} right now.`);

      const response = moodResponses[value];
      memory.addMessage(currentConversationId, 'assistant', response);

      currentAssistantContent = '';
      currentAssistantEl = null;

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const msgEl = document.createElement('div');
      msgEl.className = 'message assistant';
      msgEl.innerHTML = `
        <div class="message-avatar">
          <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
            <rect x="2" y="2" width="24" height="24" rx="7" stroke="currentColor" stroke-width="1.5" fill="none"/>
            <path d="M14 8V20M8 14H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="message-content">${engine.parseMarkdown(response)}</div>
          <div class="message-meta"><span class="message-time">${timeStr}</span></div>
        </div>
      `;
      messagesContainer.appendChild(msgEl);

      renderConversationList();
      scrollToBottom();
    }, 500);
  }

  function populateSettingsPanel() {
    $('runtime-model').textContent = runtimeConfig?.active_model || runtimeConfig?.model_id || RECALL_CONFIG.MODEL || '—';
    if (runtimeConfig?.engine) {
      $('runtime-model').textContent += ` (${runtimeConfig.engine})`;
    }
    $('runtime-endpoint').textContent = apiUrl('/v1/chat/completions');
    $('runtime-stream').textContent = RECALL_CONFIG.STREAM ? 'Enabled' : 'Disabled';
    $('runtime-temp').textContent = RECALL_CONFIG.TEMPERATURE || '—';

    $('mem-conversations').textContent = memory.getConversationCount();
    $('mem-moods').textContent = (memory.moods || []).length;
    $('mem-themes').textContent = (memory.profile?.themes || []).length;

    prefMaxTokens.value = RECALL_CONFIG.MAX_TOKENS;
    prefTemperature.value = RECALL_CONFIG.TEMPERATURE;
    prefTerminology.value = memory.profile?.preferences?.terminology || 'lay';
    prefResponseLength.value = memory.profile?.preferences?.response_length || 'balanced';
    prefTone.value = memory.profile?.preferences?.tone || 'supportive';
  }

  function updateFeatureVisibility() {
    const features = RECALL_CONFIG.ENABLED_FEATURES || {};
    if (developerAccessSection) {
      developerAccessSection.style.display = features.apiKeysEnabled ? '' : 'none';
    }
    if (moodCheckBtn) {
      moodCheckBtn.style.display = features.moodCheckEnabled === false ? 'none' : '';
    }
  }

  async function refreshHealthStatus() {
    try {
      const response = await fetch(apiUrl('/health'), { credentials: 'include' });
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      const payload = await response.json();
      if (payload.engine === 'mock') {
        setStatus('degraded', 'MedBrief AI is running in demo mode');
      } else if (payload.status !== 'healthy') {
        setStatus('degraded', 'Service temporarily degraded - please retry');
      } else {
        setStatus('ready');
      }
    } catch {
      setStatus('degraded', 'Health check unavailable');
    }
  }

  async function loadRuntimeConfig() {
    try {
      const staticRuntime = window.MedBriefRuntime
        ? await window.MedBriefRuntime.load()
        : {};

      RECALL_CONFIG.MODEL = staticRuntime.defaultModel || RECALL_CONFIG.MODEL;
      RECALL_CONFIG.API_ENDPOINT = apiUrl('/v1/chat/completions');
      RECALL_CONFIG.STREAM = staticRuntime.stream ?? RECALL_CONFIG.STREAM;
      RECALL_CONFIG.ENABLED_FEATURES = {
        ...RECALL_CONFIG.ENABLED_FEATURES,
        ...(staticRuntime.enabledFeatures || {})
      };

      if (!localStorage.getItem('recall_max_tokens') && staticRuntime.maxTokensDefault) {
        RECALL_CONFIG.MAX_TOKENS = staticRuntime.maxTokensDefault;
      }
      if (!localStorage.getItem('recall_temperature') && typeof staticRuntime.temperatureDefault === 'number') {
        RECALL_CONFIG.TEMPERATURE = staticRuntime.temperatureDefault;
      }

      const response = await fetch(apiUrl('/api/config'), { credentials: 'include' });
      if (response.ok) {
        runtimeConfig = await response.json();
        RECALL_CONFIG.MODEL = runtimeConfig.active_model || runtimeConfig.model_id || RECALL_CONFIG.MODEL;
        RECALL_CONFIG.STREAM = runtimeConfig.stream_default ?? RECALL_CONFIG.STREAM;
        RECALL_CONFIG.ENABLED_FEATURES = {
          ...RECALL_CONFIG.ENABLED_FEATURES,
          ...(runtimeConfig.frontend_features || {})
        };
        if (!localStorage.getItem('recall_max_tokens') && runtimeConfig?.default_generation?.max_new_tokens) {
          RECALL_CONFIG.MAX_TOKENS = runtimeConfig.default_generation.max_new_tokens;
        }
        if (!localStorage.getItem('recall_temperature') && typeof runtimeConfig?.default_generation?.temperature === 'number') {
          RECALL_CONFIG.TEMPERATURE = runtimeConfig.default_generation.temperature;
        }
      }
    } catch {
      RECALL_CONFIG.API_ENDPOINT = apiUrl('/v1/chat/completions');
    } finally {
      updateFeatureVisibility();
      prefMaxTokens.value = RECALL_CONFIG.MAX_TOKENS;
      prefTemperature.value = RECALL_CONFIG.TEMPERATURE;
      await refreshHealthStatus();
    }
  }

  function loadSavedApiKey() {
    if (!RECALL_CONFIG.ENABLED_FEATURES?.apiKeysEnabled) {
      RECALL_CONFIG.API_KEY = '';
      currentApiKeyId = '';
      apiKeyDisplay.style.display = 'none';
      revokeApiKeyBtn.style.display = 'none';
      return;
    }

    const savedKey = localStorage.getItem('recall_api_key');
    if (savedKey) {
      RECALL_CONFIG.API_KEY = savedKey;
      apiKeyValue.textContent = savedKey;
      apiKeyDisplay.style.display = 'block';
      revokeApiKeyBtn.style.display = 'inline-flex';
      generateApiKeyBtn.textContent = 'Regenerate API Key';
    }
  }

  async function handleGenerateApiKey() {
    if (!RECALL_CONFIG.ENABLED_FEATURES?.apiKeysEnabled) {
      appendSystemMessage('Developer API keys are disabled in this release of MedBrief AI.');
      return;
    }

    try {
      const response = await fetch(apiUrl('/api/keys'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: 'MedBrief UI key' })
      });
      if (!response.ok) {
        throw new Error('Key generation failed');
      }
      const payload = await response.json();
      const key = payload.api_key || '';
      currentApiKeyId = payload.record?.id || '';
      if (currentApiKeyId) {
        localStorage.setItem('recall_api_key_id', currentApiKeyId);
      }
      localStorage.setItem('recall_api_key', key);
      RECALL_CONFIG.API_KEY = key;
      apiKeyValue.textContent = key;
      apiKeyDisplay.style.display = 'block';
      revokeApiKeyBtn.style.display = 'inline-flex';
      generateApiKeyBtn.textContent = 'Regenerate API Key';
    } catch {
      appendSystemMessage('Unable to generate an API key right now.');
    }
  }

  async function handleRevokeApiKey() {
    if (!RECALL_CONFIG.ENABLED_FEATURES?.apiKeysEnabled) {
      appendSystemMessage('Developer API keys are disabled in this release of MedBrief AI.');
      return;
    }

    if (!currentApiKeyId) {
      localStorage.removeItem('recall_api_key');
      localStorage.removeItem('recall_api_key_id');
      RECALL_CONFIG.API_KEY = '';
      apiKeyValue.textContent = '';
      apiKeyDisplay.style.display = 'none';
      revokeApiKeyBtn.style.display = 'none';
      generateApiKeyBtn.textContent = 'Generate API Key';
      return;
    }
    try {
      const response = await fetch(apiUrl(`/api/keys/${currentApiKeyId}`), { method: 'DELETE', credentials: 'include' });
      if (!response.ok) {
        throw new Error('Key revoke failed');
      }
      currentApiKeyId = '';
      localStorage.removeItem('recall_api_key');
      localStorage.removeItem('recall_api_key_id');
      RECALL_CONFIG.API_KEY = '';
      apiKeyValue.textContent = '';
      apiKeyDisplay.style.display = 'none';
      revokeApiKeyBtn.style.display = 'none';
      generateApiKeyBtn.textContent = 'Generate API Key';
    } catch {
      appendSystemMessage('Unable to revoke the API key right now.');
    }
  }

  function handleCopyApiKey() {
    const key = apiKeyValue.textContent;
    if (!key) return;
    navigator.clipboard.writeText(key).then(() => {
      const original = apiKeyCopy.innerHTML;
      apiKeyCopy.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 7L6 10L11 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      setTimeout(() => { apiKeyCopy.innerHTML = original; }, 1500);
    });
  }

  function handleClearMemory() {
    if (confirm('This will permanently delete all conversations, mood data, and saved themes. Continue?')) {
      fetch(apiUrl(`/v1/user/${memory.getUserId()}`), { method: 'DELETE', credentials: 'include' }).catch(() => {});
      memory.clearAll();
      startNewConversation();
      populateSettingsPanel();
    }
  }

  function loadPreferences() {
    const savedTokens = localStorage.getItem('recall_max_tokens');
    const savedTemp = localStorage.getItem('recall_temperature');
    if (savedTokens) {
      const parsedTokens = parseInt(savedTokens, 10);
      const clampedTokens = Number.isFinite(parsedTokens) ? Math.min(Math.max(parsedTokens, 32), 2000) : RECALL_CONFIG.MAX_TOKENS;
      RECALL_CONFIG.MAX_TOKENS = clampedTokens;
      prefMaxTokens.value = clampedTokens;
      localStorage.setItem('recall_max_tokens', String(clampedTokens));
    }
    if (savedTemp) {
      const parsedTemp = parseFloat(savedTemp);
      const clampedTemp = Number.isFinite(parsedTemp) ? Math.min(Math.max(parsedTemp, 0), 2) : RECALL_CONFIG.TEMPERATURE;
      RECALL_CONFIG.TEMPERATURE = clampedTemp;
      prefTemperature.value = clampedTemp;
      localStorage.setItem('recall_temperature', String(clampedTemp));
    }
    prefTerminology.value = memory.profile?.preferences?.terminology || 'lay';
    prefResponseLength.value = memory.profile?.preferences?.response_length || 'balanced';
    prefTone.value = memory.profile?.preferences?.tone || 'supportive';
  }

  function generateSecureKey(length) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    const values = new Uint8Array(length);
    crypto.getRandomValues(values);
    for (let i = 0; i < length; i++) {
      result += chars[values[i] % chars.length];
    }
    return result;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  init().finally(() => {
    messageInput.focus();
  });
});
