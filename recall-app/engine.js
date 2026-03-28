class RecallEngine {
  constructor(memory) {
    this.memory = memory;
    this.abortController = null;
    this.isGenerating = false;
  }

  detectCrisis(text) {
    const lower = text.toLowerCase();
    return RECALL_CONFIG.CRISIS_KEYWORDS.some(kw => lower.includes(kw));
  }

  buildMessages(conversationId) {
    const messages = [];

    messages.push({
      role: 'system',
      content: RECALL_CONFIG.SYSTEM_PROMPT
    });

    const memoryContext = this.memory.buildMemoryContext();
    if (memoryContext) {
      messages.push({
        role: 'system',
        content: `[User Context — reference gently when relevant, never fabricate]\n${memoryContext}`
      });
    }

    const contextWindow = this.memory.getContextWindow(
      conversationId,
      RECALL_CONFIG.MAX_CONTEXT_MESSAGES
    );
    messages.push(...contextWindow);

    return messages;
  }

  async sendMessage(conversationId, userMessage, onChunk, onComplete, onError) {
    if (this.isGenerating) {
      this.cancel();
    }

    this.isGenerating = true;
    this.abortController = new AbortController();

    const isCrisis = this.detectCrisis(userMessage);

    this.memory.addMessage(conversationId, 'user', userMessage);

    const messages = this.buildMessages(conversationId);

    if (isCrisis) {
      messages.push({
        role: 'system',
        content: 'SAFETY ALERT: The user may be expressing crisis-level distress. Follow the crisis protocol immediately. Be gentle, direct, and provide emergency resources. Do not continue casual exploration.'
      });
    }

    if (!RECALL_CONFIG.API_KEY || RECALL_CONFIG.API_KEY === '') {
      onError(new Error('401'));
      return;
    }

    try {
      if (RECALL_CONFIG.STREAM) {
        await this.streamRequest(messages, conversationId, isCrisis, onChunk, onComplete, onError);
      } else {
        await this.standardRequest(messages, conversationId, isCrisis, onChunk, onComplete, onError);
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        this.isGenerating = false;
        return;
      }
      console.error('Engine error:', error);
      onError(error);
    }
  }

  async streamRequest(messages, conversationId, isCrisis, onChunk, onComplete, onError) {
    const response = await fetch(RECALL_CONFIG.API_ENDPOINT, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${RECALL_CONFIG.API_KEY}`
      },
      body: JSON.stringify({
        model: RECALL_CONFIG.MODEL,
        messages,
        max_tokens: RECALL_CONFIG.MAX_TOKENS,
        temperature: RECALL_CONFIG.TEMPERATURE,
        stream: true
      }),
      signal: this.abortController.signal
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
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

    let finalContent = fullContent;
    if (isCrisis && !fullContent.includes('988')) {
      finalContent += RECALL_CONFIG.CRISIS_RESPONSE_ADDENDUM;
      onChunk(RECALL_CONFIG.CRISIS_RESPONSE_ADDENDUM);
    }

    this.memory.addMessage(conversationId, 'assistant', finalContent);
    this.isGenerating = false;
    onComplete(finalContent);
  }

  async standardRequest(messages, conversationId, isCrisis, onChunk, onComplete, onError) {
    const response = await fetch(RECALL_CONFIG.API_ENDPOINT, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${RECALL_CONFIG.API_KEY}`
      },
      body: JSON.stringify({
        model: RECALL_CONFIG.MODEL,
        messages,
        max_tokens: RECALL_CONFIG.MAX_TOKENS,
        temperature: RECALL_CONFIG.TEMPERATURE,
        stream: false
      }),
      signal: this.abortController.signal
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    let content = data.choices?.[0]?.message?.content || 'I wasn\'t able to form a clear response. Could you try rephrasing that?';

    if (isCrisis && !content.includes('988')) {
      content += RECALL_CONFIG.CRISIS_RESPONSE_ADDENDUM;
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

  generateFallbackResponse(userMessage, isCrisis) {
    if (isCrisis) {
      return `I hear you, and I want you to know that what you're feeling matters. You don't have to go through this alone.

Please reach out to someone who can help right now:

- **988 Suicide & Crisis Lifeline** — Call or text **988** (available 24/7)
- **Crisis Text Line** — Text **HOME** to **741741**
- **Emergency Services** — Call **911** if you're in immediate danger

I'm here to listen, but a trained professional can provide the support you need right now. Your safety comes first.`;
    }

    const lower = userMessage.toLowerCase();
    const responses = this.getFallbackResponseSet();

    if (lower.includes('portfolio') || lower.includes('project') || lower.includes('work') || lower.includes('feature')) {
      return responses.portfolio;
    }
    if (lower.includes('health') || lower.includes('science') || lower.includes('medical') || lower.includes('nervous system')) {
      return responses.health;
    }
    if (lower.includes('organize') || lower.includes('plan') || lower.includes('decision') || lower.includes('structure')) {
      return responses.utility;
    }
    if (lower.includes('overwhelm') || lower.includes('too much') || lower.includes('can\'t handle')) {
      return responses.overwhelmed;
    }
    if (lower.includes('anxious') || lower.includes('anxiety') || lower.includes('worried') || lower.includes('nervous')) {
      return responses.anxious;
    }
    if (lower.includes('sad') || lower.includes('depressed') || lower.includes('down') || lower.includes('unhappy')) {
      return responses.sad;
    }
    if (lower.includes('sleep') || lower.includes('insomnia') || lower.includes('racing thoughts') || lower.includes('can\'t sleep')) {
      return responses.sleep;
    }
    if (lower.includes('procrastinat') || lower.includes('avoidance') || lower.includes('putting off')) {
      return responses.procrastination;
    }
    if (lower.includes('lonely') || lower.includes('alone') || lower.includes('isolated') || lower.includes('no one')) {
      return responses.lonely;
    }
    if (lower.includes('angry') || lower.includes('frustrated') || lower.includes('furious') || lower.includes('irritated')) {
      return responses.angry;
    }
    if (lower.includes('guilt') || lower.includes('guilty') || lower.includes('shame') || lower.includes('ashamed')) {
      return responses.guilt;
    }
    if (lower.includes('burnout') || lower.includes('burnt out') || lower.includes('exhausted') || lower.includes('drained')) {
      return responses.burnout;
    }
    if (lower.includes('pattern') || lower.includes('cycle') || lower.includes('keep doing') || lower.includes('same thing')) {
      return responses.pattern;
    }
    if (lower.includes('perfecti') || lower.includes('not good enough') || lower.includes('never enough')) {
      return responses.perfectionism;
    }

    return responses.general;
  }

  getFallbackResponseSet() {
    return {
      portfolio: `I'd be happy to share more about the portfolio and the work featured here. 

The projects focus on blending thoughtful design with practical utility, specifically in the areas of healthcare innovation, AI, and digital wellness. 

Is there a specific project or feature you'd like me to explain?`,

      health: `That's a great question. In general, physical and mental health are tightly connected — for example, prolonged stress can actively change how the nervous system regulates itself, leading to exhaustion or elevated anxiety.

I can provide clear, educational explanations about health concepts like this. However, please remember I'm an AI, so none of this is medical advice! What part of this topic are you most curious about?`,

      utility: `I find that organizing thoughts is half the battle when making a big decision or starting a complex plan. 

It usually helps to break things down into three buckets: 
1. What you know for sure
2. What you still need to figure out
3. What you can't control

Would it be helpful to start dividing your thoughts into those categories?`,

      overwhelmed: `That sounds like a lot to carry. When everything piles up at once, even small things can feel heavier than they should.

It might help to separate what's urgent from what's just noisy. What feels like the most pressing thing right now — not the biggest problem, just the one taking up the most mental space?`,

      anxious: `Anxiety has a way of making everything feel urgent and uncertain at the same time. That tension between wanting control and feeling like you don't have it — it's exhausting.

Sometimes it helps to ask: what's the actual worst-case scenario here, and how likely is it really? Not to dismiss the feeling, but to give your mind something concrete to work with instead of the spiral.

What's the worry that keeps circling back the most?`,

      sad: `I'm sorry you're feeling this way. Sadness doesn't always need a dramatic reason — sometimes it just settles in, and that's valid too.

You don't need to fix it right now. But if you're open to it, it can help to notice: is this sadness about something specific, or does it feel more like a general heaviness? That distinction sometimes points toward what might help.`,

      sleep: `Racing thoughts at night are one of the hardest things to manage, because the quieter the room gets, the louder your mind becomes.

One thing that sometimes helps is a "thought dump" — writing out everything cycling through your head for 5 minutes before bed. Not organized, not pretty. Just getting it out of your head and onto paper so your brain can let go of holding onto it. Would you be willing to try that tonight?`,

      procrastination: `Procrastination usually isn't about laziness — it's often about fear. Fear of doing it wrong, fear of how much energy it'll take, fear of what finishing (or not finishing) means.

The cycle tends to look like: avoid → feel relief → then guilt → then more pressure → more avoidance. Sound familiar?

What if you set a timer for 10 minutes and just started the easiest part? Not to finish — just to break the stillness.`,

      lonely: `Feeling alone — even when people are around — is one of the heaviest experiences. It's not just about having company, it's about feeling genuinely seen and connected.

Sometimes loneliness becomes a loop: you withdraw because connection feels hard, and then the withdrawal makes you feel more alone. Does that ring true for you?

Even one small moment of real connection can start to shift that. Is there someone — even casually — you feel relatively safe reaching out to?`,

      angry: `Anger often shows up when something important to you has been crossed — a boundary, an expectation, a sense of fairness. It's trying to tell you something.

The tricky part is that anger can be loud enough to drown out the softer feeling underneath it. Sometimes under anger there's hurt, disappointment, or feeling unseen.

What happened that brought this up? I'd like to understand what's underneath it.`,

      guilt: `Guilt can be useful when it points toward something you genuinely want to repair. But a lot of the time, guilt becomes disproportionate — it makes you feel responsible for things that aren't really yours to carry.

It might help to ask yourself: if a close friend did the exact thing I feel guilty about, would I judge them this harshly? Usually, the answer reveals how much extra weight you're putting on yourself.

What's the guilt about, if you're comfortable sharing?`,

      burnout: `Burnout isn't just being tired — it's being tired of being tired. It's when the things that used to motivate you start feeling empty, and rest alone doesn't seem to fix it.

The hard part is that burnout often makes you feel like you can't afford to stop, even though stopping is exactly what you need.

What would actual recovery look like for you today — not a vacation fantasy, but one realistic thing that would give you even a small amount of genuine rest?`,

      pattern: `Noticing a pattern is actually a really important step. Most people stay stuck in loops because they can't see them clearly — the fact that you're recognizing it means you're already starting to step outside of it.

The typical cycle usually has a trigger, an emotional response, a habitual behavior, and a consequence that sets up the next round. Can you walk me through what yours looks like? What usually happens first?`,

      perfectionism: `Perfectionism often disguises itself as high standards, but underneath it's usually driven by a fear — of failure, of judgment, of not being enough. The irony is that it paralyzes you into doing less, not more.

The loop usually looks like: set impossibly high bar → feel unable to meet it → avoid starting → feel worse about not starting. Does that sound like what's happening?

What if "good enough" was actually an act of courage instead of a compromise?`,

      general: `Thank you for sharing that. I want to make sure I understand what you're going through.

Could you tell me a bit more about what's on your mind? I'd like to get a clearer picture so I can be more helpful. What feels most important to talk about right now?`
    };
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
