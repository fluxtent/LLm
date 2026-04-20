const DEFAULT_MEDBRIEF_RUNTIME_CONFIG = {
  apiBaseUrl: 'http://127.0.0.1:8001',
  defaultModel: 'phi3:mini',
  stream: true,
  enabledFeatures: {
    apiKeysEnabled: false,
    moodCheckEnabled: true,
    memoryInsightsEnabled: true,
    feedbackEnabled: true,
    profileEnabled: true,
    modeTooltipsEnabled: true
  },
  maxTokensDefault: 80,
  temperatureDefault: 0.25
};

window.MedBriefRuntime = {
  _loaded: null,
  _config: { ...DEFAULT_MEDBRIEF_RUNTIME_CONFIG },

  get() {
    return {
      ...DEFAULT_MEDBRIEF_RUNTIME_CONFIG,
      ...this._config,
      enabledFeatures: {
        ...DEFAULT_MEDBRIEF_RUNTIME_CONFIG.enabledFeatures,
        ...(this._config.enabledFeatures || {})
      }
    };
  },

  apiUrl(path) {
    const runtime = this.get();
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    const base = (runtime.apiBaseUrl || '').trim().replace(/\/+$/, '');
    return base ? `${base}${normalizedPath}` : normalizedPath;
  },

  async load() {
    if (this._loaded) {
      return this._loaded;
    }

    this._loaded = fetch('./runtime-config.json', {
      cache: 'no-store'
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Runtime config load failed: ${response.status}`);
        }
        return response.json();
      })
      .then(config => {
        this._config = {
          ...this._config,
          ...config,
          enabledFeatures: {
            ...this._config.enabledFeatures,
            ...(config.enabledFeatures || {})
          }
        };
        return this.get();
      })
      .catch(() => this.get());

    return this._loaded;
  }
};
