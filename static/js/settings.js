/**
 * settings.js — LLM 编排设置面板。
 */

const LLMSettings = {
    _fallbackCounter: 0,

    /** 初始化：绑定事件 */
    init() {
        document.getElementById('btn-settings').addEventListener('click', () => {
            this._loadSettings();
            document.getElementById('modal-settings').style.display = 'flex';
        });
        document.getElementById('btn-close-settings').addEventListener('click', () => {
            document.getElementById('modal-settings').style.display = 'none';
        });
        document.getElementById('btn-save-settings').addEventListener('click', () => {
            this._saveSettings();
        });
        document.getElementById('btn-add-fallback').addEventListener('click', () => {
            this._addFallbackRow();
        });

        // Temperature 滑块联动
        const tempSlider = document.getElementById('llm-primary-temperature');
        tempSlider.addEventListener('input', () => {
            document.getElementById('llm-temp-val').textContent =
                parseFloat(tempSlider.value).toFixed(2);
        });
    },

    /** 从服务器加载 LLM 配置 */
    _loadSettings() {
        fetch('/api/config/llm')
            .then(r => r.json())
            .then(cfg => {
                if (cfg.error) { alert('加载配置失败: ' + cfg.error); return; }

                // 主力模型
                const p = cfg.primary || {};
                document.getElementById('llm-primary-provider').value = p.provider || 'anthropic';
                document.getElementById('llm-primary-model').value = p.model || '';
                document.getElementById('llm-primary-api-key').value = p.api_key || '';
                document.getElementById('llm-primary-base-url').value = p.base_url || '';
                document.getElementById('llm-primary-temperature').value = p.temperature ?? 0.1;
                document.getElementById('llm-temp-val').textContent = (p.temperature ?? 0.1).toFixed(2);
                document.getElementById('llm-primary-max-tokens').value = p.max_tokens || 200;
                document.getElementById('llm-primary-timeout').value = p.timeout_seconds || 15;

                // 策略
                document.getElementById('llm-call-frequency').value = cfg.call_frequency || 'every';
                document.getElementById('llm-min-decisions').value = cfg.min_llm_decisions_per_hand ?? 1;
                document.getElementById('llm-context-window').value = cfg.context_window_hands ?? 5;

                // 高级
                document.getElementById('llm-prompt-caching').checked = cfg.enable_prompt_caching !== false;
                document.getElementById('llm-commentary').checked = cfg.enable_commentary === true;
                document.getElementById('llm-advisor').checked = cfg.enable_advisor === true;

                // 降级链
                this._renderFallbacks(cfg.fallbacks || []);
            })
            .catch(e => alert('加载配置失败: ' + e));
    },

    /** 渲染降级链 */
    _renderFallbacks(fallbacks) {
        const container = document.getElementById('fallback-list');
        container.innerHTML = '';
        this._fallbackCounter = 0;

        fallbacks.forEach(fb => {
            this._addFallbackRow(fb);
        });

        // 如果为空，显示提示
        if (fallbacks.length === 0) {
            container.innerHTML = '<p style="color:#888;font-size:13px;">暂无降级模型（将使用规则引擎作为最终兜底）</p>';
        }
    },

    /** 添加一个降级链行 */
    _addFallbackRow(data = null) {
        const container = document.getElementById('fallback-list');
        // 清除空提示
        const hint = container.querySelector('.fallback-hint');
        if (hint) hint.remove();

        const idx = ++this._fallbackCounter;
        const row = document.createElement('div');
        row.className = 'fallback-row';
        row.innerHTML = `
            <span class="fallback-arrow">↳</span>
            <select class="fb-provider">
                <option value="anthropic" ${(data?.provider || 'anthropic') === 'anthropic' ? 'selected' : ''}>Anthropic</option>
                <option value="openai" ${data?.provider === 'openai' ? 'selected' : ''}>OpenAI</option>
                <option value="ollama" ${data?.provider === 'ollama' ? 'selected' : ''}>Ollama</option>
            </select>
            <input type="text" class="fb-model" placeholder="模型 ID" value="${data?.model || ''}">
            <input type="number" class="fb-timeout" placeholder="超时(秒)" value="${data?.timeout_seconds || 10}" min="3" max="60" style="width:60px;">
            <button class="btn btn-sm btn-remove-fallback" title="移除">✕</button>
        `;
        row.querySelector('.btn-remove-fallback').addEventListener('click', () => {
            row.remove();
            if (container.querySelectorAll('.fallback-row').length === 0) {
                container.innerHTML = '<p class="fallback-hint" style="color:#888;font-size:13px;">暂无降级模型（将使用规则引擎作为最终兜底）</p>';
            }
        });
        container.appendChild(row);
    },

    /** 保存 LLM 配置 */
    _saveSettings() {
        const primaryApiKey = document.getElementById('llm-primary-api-key').value;

        // 收集降级链
        const fallbacks = [];
        document.querySelectorAll('.fallback-row').forEach(row => {
            const provider = row.querySelector('.fb-provider').value;
            const model = row.querySelector('.fb-model').value.trim();
            const timeout = parseInt(row.querySelector('.fb-timeout').value) || 10;
            if (model) {
                fallbacks.push({ provider, model, timeout_seconds: timeout });
            }
        });

        const cfg = {
            primary: {
                provider: document.getElementById('llm-primary-provider').value,
                model: document.getElementById('llm-primary-model').value.trim(),
                api_key: primaryApiKey,  // 空字符串表示不更新
                base_url: document.getElementById('llm-primary-base-url').value.trim(),
                temperature: parseFloat(document.getElementById('llm-primary-temperature').value),
                max_tokens: parseInt(document.getElementById('llm-primary-max-tokens').value) || 200,
                timeout_seconds: parseInt(document.getElementById('llm-primary-timeout').value) || 15,
            },
            fallbacks,
            call_frequency: document.getElementById('llm-call-frequency').value,
            min_llm_decisions_per_hand: parseInt(document.getElementById('llm-min-decisions').value) || 1,
            context_window_hands: parseInt(document.getElementById('llm-context-window').value) || 5,
            enable_prompt_caching: document.getElementById('llm-prompt-caching').checked,
            enable_commentary: document.getElementById('llm-commentary').checked,
            enable_advisor: document.getElementById('llm-advisor').checked,
        };

        fetch('/api/config/llm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg),
        })
        .then(r => r.json())
        .then(result => {
            if (result.error) {
                alert('保存失败: ' + result.error);
            } else {
                document.getElementById('modal-settings').style.display = 'none';
                alert('✅ LLM 配置已保存！新游戏将使用新配置。');
            }
        })
        .catch(e => alert('保存失败: ' + e));
    },
};

// 在 App 初始化后初始化 Settings
document.addEventListener('DOMContentLoaded', () => LLMSettings.init());
