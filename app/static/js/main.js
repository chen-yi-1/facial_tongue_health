document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('upload-form');
  const resultDiv = document.getElementById('result');
  const labelP = document.getElementById('label');
  const probDiv = document.getElementById('probabilities');
  const msgDiv = document.getElementById('message');
  const submitBtn = document.getElementById('submit-btn');
  const faceArea = document.getElementById('face-area');
  const tongueArea = document.getElementById('tongue-area');
  const facePreviewImg = document.getElementById('face-preview');
  const tonguePreviewImg = document.getElementById('tongue-preview');
  const faceInput = document.getElementById('face_image');
  const tongueInput = document.getElementById('tongue_image');
  const progressBar = document.getElementById('analyze-progress');
  const adviceSection = document.getElementById('advice-section');
  const ruleGrid = document.getElementById('rule-grid');
  const aiGrid = document.getElementById('ai-grid');
  const tabRule = document.getElementById('tab-rule');
  const tabAi = document.getElementById('tab-ai');

  let faceObjectUrl = null;
  let tongueObjectUrl = null;

  function setupDragDrop(area, input, previewImg, objUrlRef) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      area.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      area.addEventListener(eventName, () => area.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      area.addEventListener(eventName, () => area.classList.remove('dragover'), false);
    });

    area.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        input.files = files;
        input.dispatchEvent(new Event('change'));
      }
    }, false);
  }

  function handleFileSelect(input, area, previewImg, objUrlRef) {
    const file = input.files && input.files[0];
    if (!file) {
      area.classList.remove('has-file');
      return;
    }

    if (objUrlRef.value) URL.revokeObjectURL(objUrlRef.value);
    objUrlRef.value = URL.createObjectURL(file);
    previewImg.src = objUrlRef.value;
    area.classList.add('has-file');
  }

  faceInput.addEventListener('change', () => {
    handleFileSelect(faceInput, faceArea, facePreviewImg, { value: faceObjectUrl });
  });

  tongueInput.addEventListener('change', () => {
    handleFileSelect(tongueInput, tongueArea, tonguePreviewImg, { value: tongueObjectUrl });
  });

  setupDragDrop(faceArea, faceInput, facePreviewImg, { value: faceObjectUrl });
  setupDragDrop(tongueArea, tongueInput, tonguePreviewImg, { value: tongueObjectUrl });

  function showLoading() {
    progressBar.classList.remove('animate-progress');
    progressBar.style.width = '0';
    void progressBar.offsetWidth;
    progressBar.style.opacity = '1';
    progressBar.classList.add('animate-progress');
  }

  function hideLoading() {
    progressBar.classList.remove('animate-progress');
    progressBar.style.transition = 'width 0.3s ease-out';
    progressBar.style.width = '100%';
    setTimeout(() => {
      progressBar.style.transition = 'opacity 0.3s ease-out';
      progressBar.style.opacity = '0';
    }, 300);
  }

  function animateProbBars() {
    const fills = document.querySelectorAll('.prob-fill');
    fills.forEach((fill, index) => {
      const targetWidth = fill.dataset.width;
      setTimeout(() => {
        fill.style.width = targetWidth + '%';
      }, index * 150);
    });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();

    document.getElementById('panel-empty').classList.add('hidden');
    msgDiv.textContent = '正在分析，请稍候...';
    msgDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    probDiv.innerHTML = '';

    const skeletonHTML = `
      <div class="prob-item" style="opacity: 1; transform: none;">
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-bar"></div>
      </div>
      <div class="prob-item" style="opacity: 1; transform: none;">
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-bar"></div>
      </div>
      <div class="prob-item" style="opacity: 1; transform: none;">
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-bar"></div>
      </div>
    `;
    probDiv.innerHTML = skeletonHTML;
    resultDiv.classList.remove('hidden');

    const formData = new FormData(form);

    submitBtn.disabled = true;
    submitBtn.classList.add('loading');

    showLoading();

    fetch('/predict', {
      method: 'POST',
      body: formData
    }).then(resp => resp.json())
      .then(data => {
        hideLoading();
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');

        if (!data.success) {
          msgDiv.textContent = data.msg || '识别失败，请重试。';
          resultDiv.classList.add('hidden');
          return;
        }

        msgDiv.classList.add('hidden');
        labelP.textContent = data.label;

        // 渲染生活建议
        if (data.advice && data.advice.rule) {
          renderAdvice('rule', data.advice.rule);
          adviceSection.classList.remove('hidden');
          // 异步获取 AI 建议
          fetchAiAdvice(data.label, data.probabilities);
        } else {
          adviceSection.classList.add('hidden');
        }

        probDiv.innerHTML = '';
        const probs = data.probabilities || {};

        Object.keys(probs).forEach((key, index) => {
          const value = (probs[key] * 100).toFixed(2);

          const item = document.createElement('div');
          item.className = 'prob-item';
          item.style.animationDelay = `${index * 100}ms`;

          item.innerHTML = `
            <div class="prob-label-row">
              <span class="prob-label">${key}</span>
              <span class="prob-value">${value}%</span>
            </div>
            <div class="prob-bar">
              <div class="prob-fill" data-width="${value}"></div>
            </div>
          `;

          probDiv.appendChild(item);
        });

        setTimeout(animateProbBars, 100);
      })
      .catch(err => {
        console.error(err);
        hideLoading();
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
        msgDiv.textContent = '请求异常，请检查后端是否启动。';
        resultDiv.classList.add('hidden');
      });
  });

  /* ========== 生活建议 ========== */

  const ADVICE_ICONS = {
    '饮食调理': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>',
    '作息建议': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    '运动指导': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    '中医调理': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4.8 2.3A18 18 0 0 1 12 2a18 18 0 0 1 7.2.3"/><path d="M12 2v20"/><path d="M12 12l-2-2"/><path d="M12 12l2-2"/></svg>',
    '心理调节': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
  };

  function renderAdvice(type, adviceData) {
    const grid = type === 'rule' ? ruleGrid : aiGrid;
    const dimensions = ['饮食调理', '作息建议', '运动指导', '中医调理', '心理调节'];
    grid.innerHTML = '';

    dimensions.forEach((dim) => {
      const text = adviceData[dim] || '暂无建议';
      const card = document.createElement('div');
      card.className = 'advice-card';
      card.innerHTML = `
        <div class="advice-card-header">
          <span class="advice-card-icon" style="color: var(--color-primary);">${ADVICE_ICONS[dim] || ''}</span>
          <span class="advice-card-title">${dim}</span>
        </div>
        <p class="advice-card-text">${text}</p>
      `;
      grid.appendChild(card);
    });
  }

  function fetchAiAdvice(label, probabilities) {
    tabAi.classList.add('loading');
    tabAi.disabled = true;

    fetch('/advice/ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, probabilities }),
    })
      .then((resp) => resp.json())
      .then((data) => {
        tabAi.classList.remove('loading');
        if (data.success && data.advice && data.advice.ai) {
          renderAdvice('ai', data.advice.ai);
          tabAi.disabled = false;
        } else {
          tabAi.title = 'AI 建议暂不可用';
        }
      })
      .catch(() => {
        tabAi.classList.remove('loading');
        tabAi.title = 'AI 建议加载失败';
      });
  }

  // Tab 切换
  tabRule.addEventListener('click', () => {
    tabRule.classList.add('active');
    tabAi.classList.remove('active');
    document.getElementById('advice-rule').classList.remove('hidden');
    document.getElementById('advice-ai').classList.add('hidden');
  });

  tabAi.addEventListener('click', () => {
    if (tabAi.disabled) return;
    tabAi.classList.add('active');
    tabRule.classList.remove('active');
    document.getElementById('advice-rule').classList.add('hidden');
    document.getElementById('advice-ai').classList.remove('hidden');
  });
});