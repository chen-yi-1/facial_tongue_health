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

    msgDiv.textContent = '';
    msgDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    const formData = new FormData(form);

    submitBtn.disabled = true;
    submitBtn.classList.add('loading');

    fetch('/predict', {
      method: 'POST',
      body: formData
    }).then(resp => resp.json())
      .then(data => {
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');

        if (!data.success) {
          msgDiv.textContent = data.msg || '识别失败，请重试。';
          return;
        }

        labelP.textContent = data.label;

        probDiv.innerHTML = '';
        const probs = data.probabilities || {};
        
        Object.keys(probs).forEach(key => {
          const value = (probs[key] * 100).toFixed(2);
          
          const item = document.createElement('div');
          item.className = 'prob-item';
          
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

        resultDiv.classList.remove('hidden');
        
        setTimeout(animateProbBars, 100);
      })
      .catch(err => {
        console.error(err);
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
        msgDiv.textContent = '请求异常，请检查后端是否启动。';
      });
  });
});