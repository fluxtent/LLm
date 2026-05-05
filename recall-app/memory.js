class RecallMemory {
  constructor() {
    this.conversationsKey = 'recall_conversations';
    this.moodKey = 'recall_moods';
    this.profileKey = 'medbrief_profile';
    this.load();
  }

  load() {
    try {
      this.conversations = JSON.parse(localStorage.getItem(this.conversationsKey)) || [];
      this.moods = JSON.parse(localStorage.getItem(this.moodKey)) || [];
      this.profile = JSON.parse(localStorage.getItem(this.profileKey)) || this.defaultProfile();
      if (!this.profile.user_id) {
        this.profile.user_id = this.generateId();
      }
      if (!this.profile.preferences) {
        this.profile.preferences = this.defaultProfile().preferences;
      }
      if (!Array.isArray(this.profile.session_history)) {
        this.profile.session_history = [];
      }
      if (!Array.isArray(this.profile.medical_context)) {
        this.profile.medical_context = [];
      }
      if (!Array.isArray(this.profile.recurring_topics)) {
        this.profile.recurring_topics = [];
      }
      if (!Array.isArray(this.profile.mood_history)) {
        this.profile.mood_history = [];
      }
    } catch {
      this.conversations = [];
      this.moods = [];
      this.profile = this.defaultProfile();
    }
  }

  defaultProfile() {
    return {
      user_id: this.generateId(),
      display_name: '',
      communication_style: '',
      medical_context: [],
      recurring_topics: [],
      session_history: [],
      mood_history: [],
      preferences: {
        terminology: 'lay',
        response_length: 'balanced',
        tone: 'supportive',
        memory_enabled: true
      },
      themes: [],
      copingPrefs: [],
      recurringStressors: [],
      patterns: [],
      goals: [],
      createdAt: Date.now()
    };
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

  getUserId() {
    return this.profile.user_id;
  }

  getUserProfile() {
    return JSON.parse(JSON.stringify(this.profile));
  }

  setServerProfile(profile) {
    if (!profile || !profile.user_id) return;
    this.profile = {
      ...this.defaultProfile(),
      ...this.profile,
      ...profile,
      preferences: {
        ...this.defaultProfile().preferences,
        ...(this.profile.preferences || {}),
        ...(profile.preferences || {})
      }
    };
    this.save();
  }

  updatePreferences(nextPreferences) {
    this.profile.preferences = {
      ...this.defaultProfile().preferences,
      ...(this.profile.preferences || {}),
      ...nextPreferences
    };
    this.save();
  }

  addSessionSummary(summary) {
    if (!summary) return;
    this.profile.session_history = [summary, ...(this.profile.session_history || []).filter(item => item !== summary)].slice(0, 12);
    this.save();
  }

  createConversation() {
    const conv = {
      id: this.generateId(),
      title: 'New conversation',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      summary: '',
      dominantMode: 'general',
      modeCounts: {
        psych: 0,
        health: 0,
        crisis: 0,
        portfolio: 0,
        general: 0
      }
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

    if (role === 'user') {
      this.updateConversationMode(conv, content);
      this.learnFromUserMessage(content);
    }

    this.save();
    return msg;
  }

  detectMode(text) {
    const lower = (text || '').toLowerCase();
    const psychKeywords = ['anxious', 'depressed', 'feeling', 'overwhelmed', 'struggling', 'therapist', 'panic', 'burnout', 'lonely', 'hurt', 'hurting', 'grief', 'lost', 'failure', 'failed', 'can\'t think', 'cant think', 'all over', 'hopeless', 'worthless', 'ruined'];
    const healthKeywords = ['symptom', 'medication', 'diagnosis', 'doctor', 'pain', 'condition', 'treatment', 'hospital', 'medical', 'health', 'headache', 'sleep'];
    const portfolioKeywords = ['medbrief', 'portfolio', 'project', 'website', 'product', 'feature'];

    if (this.detectCrisis(lower)) return 'crisis';
    if (psychKeywords.some(kw => lower.includes(kw))) return 'psych';
    if (healthKeywords.some(kw => lower.includes(kw))) return 'health';
    if (portfolioKeywords.some(kw => lower.includes(kw))) return 'portfolio';
    return 'general';
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

  updateConversationMode(conv, text) {
    const mode = this.detectMode(text);
    conv.modeCounts = conv.modeCounts || { psych: 0, health: 0, crisis: 0, portfolio: 0, general: 0 };
    conv.modeCounts[mode] = (conv.modeCounts[mode] || 0) + 1;
    conv.dominantMode = Object.entries(conv.modeCounts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'general';
  }

  learnFromUserMessage(content) {
    const lower = content.toLowerCase();
    const themeCandidates = ['anxiety', 'stress', 'sleep', 'burnout', 'relationships', 'purpose', 'grief', 'focus', 'work', 'health'];
    const patternCandidates = ['avoidance', 'perfectionism', 'shutdown', 'spiral', 'rumination', 'self-criticism'];
    const medicalCandidates = ['headache', 'fatigue', 'insomnia', 'ssri', 'burnout', 'panic', 'depression'];

    themeCandidates.forEach(theme => {
      if (lower.includes(theme)) this.addTheme(theme);
    });
    patternCandidates.forEach(pattern => {
      if (lower.includes(pattern)) this.addPattern(pattern);
    });
    medicalCandidates.forEach(term => {
      if (lower.includes(term) && !this.profile.medical_context.includes(term)) {
        this.profile.medical_context.push(term);
      }
    });

    const recurringTopics = new Set(this.profile.recurring_topics || []);
    themeCandidates.forEach(theme => {
      if (lower.includes(theme)) recurringTopics.add(theme);
    });
    this.profile.recurring_topics = Array.from(recurringTopics).slice(-12);
    this.save();
  }

  generateTitle(content) {
    const cleaned = content.replace(/\n/g, ' ').trim();
    if (cleaned.length <= 40) return cleaned;
    const truncated = cleaned.substring(0, 40);
    const lastSpace = truncated.lastIndexOf(' ');
    return (lastSpace > 20 ? truncated.substring(0, lastSpace) : truncated) + '…';
  }

  needsSummarization(convId) {
    const conv = this.getConversation(convId);
    if (!conv) return false;
    return !!this.profile.preferences.memory_enabled && conv.messages.length > (RECALL_CONFIG.MEMORY_SUMMARY_THRESHOLD || 20);
  }

  applyServerSummary(convId, summary) {
    const conv = this.getConversation(convId);
    if (!conv || !summary) return;
    conv.summary = summary;
    conv.messages = conv.messages.slice(-12);
    this.addSessionSummary(summary);
    this.save();
  }

  latestUserPrompt(convId) {
    const conv = this.getConversation(convId);
    if (!conv) return '';
    const reversed = [...conv.messages].reverse();
    return reversed.find(message => message.role === 'user')?.content || '';
  }

  getContextWindow(convId, maxMessages) {
    const conv = this.getConversation(convId);
    if (!conv) return [];
    const messages = conv.messages.slice(-maxMessages).map(m => ({ role: m.role, content: m.content }));
    if (conv.summary) {
      return [{ role: 'system', content: `Conversation summary: ${conv.summary}` }, ...messages];
    }
    return messages;
  }

  buildCrossSessionInsight() {
    if (this.conversations.length < 5) return '';
    const dominantModes = this.conversations.map(conv => conv.dominantMode || 'general');
    const modeCounts = dominantModes.reduce((acc, mode) => {
      acc[mode] = (acc[mode] || 0) + 1;
      return acc;
    }, {});
    const leadMode = Object.entries(modeCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
    const recurringThemes = (this.profile.themes || []).slice(-3);

    if (!leadMode && recurringThemes.length === 0) return '';

    let insight = `Across recent conversations, the dominant mode has been ${leadMode || 'general'}.`;
    if (recurringThemes.length > 0) {
      insight += ` Recurring themes include ${recurringThemes.join(', ')}.`;
    }
    return insight;
  }

  buildMemoryContext() {
    const parts = [];

    if (this.profile.recurring_topics?.length > 0) {
      parts.push(`Recurring topics: ${this.profile.recurring_topics.join(', ')}`);
    }
    if (this.profile.patterns?.length > 0) {
      parts.push(`Identified patterns: ${this.profile.patterns.join('; ')}`);
    }
    if (this.profile.medical_context?.length > 0) {
      parts.push(`Medical context hints: ${this.profile.medical_context.join(', ')}`);
    }
    if (this.profile.session_history?.length > 0) {
      parts.push(`Recent session summary: ${this.profile.session_history[0]}`);
    }

    const recentMoods = this.moods.slice(-7);
    if (recentMoods.length > 0) {
      const moodSummary = recentMoods.map(m =>
        `${RECALL_CONFIG.MOOD_LABELS[m.value] || m.value} (${new Date(m.timestamp).toLocaleDateString()})`
      ).join(', ');
      parts.push(`Recent mood check-ins: ${moodSummary}`);
    }

    const sessionInsight = this.buildCrossSessionInsight();
    if (sessionInsight) {
      parts.push(sessionInsight);
    }

    return parts.length > 0 ? parts.join('\n') : '';
  }

  recordMood(value) {
    this.moods.push({ value, timestamp: Date.now() });
    this.profile.mood_history = [...(this.profile.mood_history || []), { value, timestamp: Date.now() }].slice(-100);
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
    return Date.now().toString(36) + Math.random().toString(36).substring(2, 10);
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
    this.profile = this.defaultProfile();
    this.save();
  }
}
