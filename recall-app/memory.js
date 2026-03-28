class RecallMemory {
  constructor() {
    this.storageKey = 'recall_memory';
    this.conversationsKey = 'recall_conversations';
    this.moodKey = 'recall_moods';
    this.profileKey = 'recall_profile';
    this.load();
  }

  load() {
    try {
      this.conversations = JSON.parse(localStorage.getItem(this.conversationsKey)) || [];
      this.moods = JSON.parse(localStorage.getItem(this.moodKey)) || [];
      this.profile = JSON.parse(localStorage.getItem(this.profileKey)) || {
        themes: [],
        copingPrefs: [],
        recurringStressors: [],
        patterns: [],
        goals: [],
        createdAt: Date.now()
      };
    } catch {
      this.conversations = [];
      this.moods = [];
      this.profile = { themes: [], copingPrefs: [], recurringStressors: [], patterns: [], goals: [], createdAt: Date.now() };
    }
  }

  save() {
    try {
      localStorage.setItem(this.conversationsKey, JSON.stringify(this.conversations));
      localStorage.setItem(this.moodKey, JSON.stringify(this.moods));
      localStorage.setItem(this.profileKey, JSON.stringify(this.profile));
    } catch (e) {
      console.warn('Storage save failed:', e);
    }
  }

  createConversation() {
    const conv = {
      id: this.generateId(),
      title: 'New conversation',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      summary: ''
    };
    this.conversations.unshift(conv);
    this.save();
    return conv;
  }

  getConversation(id) {
    return this.conversations.find(c => c.id === id) || null;
  }

  updateConversation(id, updates) {
    const conv = this.getConversation(id);
    if (conv) {
      Object.assign(conv, updates, { updatedAt: Date.now() });
      this.save();
    }
    return conv;
  }

  deleteConversation(id) {
    this.conversations = this.conversations.filter(c => c.id !== id);
    this.save();
  }

  addMessage(convId, role, content) {
    const conv = this.getConversation(convId);
    if (!conv) return null;

    const msg = {
      id: this.generateId(),
      role,
      content,
      timestamp: Date.now()
    };
    conv.messages.push(msg);
    conv.updatedAt = Date.now();

    if (conv.messages.length === 1 && role === 'user') {
      conv.title = this.generateTitle(content);
    }

    this.save();
    return msg;
  }

  generateTitle(content) {
    const cleaned = content.replace(/\n/g, ' ').trim();
    if (cleaned.length <= 40) return cleaned;
    const truncated = cleaned.substring(0, 40);
    const lastSpace = truncated.lastIndexOf(' ');
    return (lastSpace > 20 ? truncated.substring(0, lastSpace) : truncated) + '…';
  }

  getContextWindow(convId, maxMessages) {
    const conv = this.getConversation(convId);
    if (!conv) return [];
    const messages = conv.messages.slice(-maxMessages);
    return messages.map(m => ({ role: m.role, content: m.content }));
  }

  buildMemoryContext() {
    const parts = [];

    if (this.profile.themes.length > 0) {
      parts.push(`Recurring themes: ${this.profile.themes.join(', ')}`);
    }
    if (this.profile.patterns.length > 0) {
      parts.push(`Identified patterns: ${this.profile.patterns.join('; ')}`);
    }
    if (this.profile.recurringStressors.length > 0) {
      parts.push(`Known stressors: ${this.profile.recurringStressors.join(', ')}`);
    }
    if (this.profile.goals.length > 0) {
      parts.push(`Active goals: ${this.profile.goals.join(', ')}`);
    }

    const recentMoods = this.moods.slice(-7);
    if (recentMoods.length > 0) {
      const moodSummary = recentMoods.map(m =>
        `${RECALL_CONFIG.MOOD_LABELS[m.value] || m.value} (${new Date(m.timestamp).toLocaleDateString()})`
      ).join(', ');
      parts.push(`Recent mood check-ins: ${moodSummary}`);
    }

    return parts.length > 0 ? parts.join('\n') : '';
  }

  recordMood(value) {
    this.moods.push({ value, timestamp: Date.now() });
    if (this.moods.length > 100) {
      this.moods = this.moods.slice(-100);
    }
    this.save();
  }

  addTheme(theme) {
    if (!this.profile.themes.includes(theme)) {
      this.profile.themes.push(theme);
      if (this.profile.themes.length > 15) this.profile.themes.shift();
      this.save();
    }
  }

  addPattern(pattern) {
    if (!this.profile.patterns.includes(pattern)) {
      this.profile.patterns.push(pattern);
      if (this.profile.patterns.length > 10) this.profile.patterns.shift();
      this.save();
    }
  }

  addStressor(stressor) {
    if (!this.profile.recurringStressors.includes(stressor)) {
      this.profile.recurringStressors.push(stressor);
      if (this.profile.recurringStressors.length > 10) this.profile.recurringStressors.shift();
      this.save();
    }
  }

  generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
  }

  getAllConversations() {
    return this.conversations.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  getConversationCount() {
    return this.conversations.length;
  }

  getMoodTrend(days = 7) {
    const cutoff = Date.now() - (days * 24 * 60 * 60 * 1000);
    return this.moods.filter(m => m.timestamp >= cutoff);
  }

  clearAll() {
    this.conversations = [];
    this.moods = [];
    this.profile = { themes: [], copingPrefs: [], recurringStressors: [], patterns: [], goals: [], createdAt: Date.now() };
    this.save();
  }
}
