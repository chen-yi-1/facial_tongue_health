document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('upload-form');
  const resultDiv = document.getElementById('result');
  const labelP = document.getElementById('label');
  const probDiv = document.getElementById('probabilities');
  const msgDiv = document.getElementById('message');
  const submitBtn = document.getElementById('submit-btn');
  const previewsDiv = document.getElementById('upload-previews');
  const facePreviewImg = document.getElementById('face-preview');
  const tonguePreviewImg = document.getElementById('tongue-preview');
  const faceInput = document.getElementById('face_image');
  const tongueInput = document.getElementById('tongue_image');

  // 选择文件后立刻本地预览（不等待 /predict 返回）
  let faceObjectUrl = null;
  let tongueObjectUrl = null;

  function showPreviews() {
    previewsDiv.classList.remove('hidden');
  }

  faceInput.addEventListener('change', function () {
    const file = faceInput.files && faceInput.files[0];
    if (!file) return;
    if (faceObjectUrl) URL.revokeObjectURL(faceObjectUrl);
    faceObjectUrl = URL.createObjectURL(file);
    facePreviewImg.src = faceObjectUrl;
    showPreviews();
  });

  tongueInput.addEventListener('change', function () {
    const file = tongueInput.files && tongueInput.files[0];
    if (!file) return;
    if (tongueObjectUrl) URL.revokeObjectURL(tongueObjectUrl);
    tongueObjectUrl = URL.createObjectURL(file);
    tonguePreviewImg.src = tongueObjectUrl;
    showPreviews();
  });

  form.addEventListener('submit', function (e) {
    e.preventDefault();

    msgDiv.textContent = '';
    resultDiv.classList.add('hidden');

    const formData = new FormData(form);

    submitBtn.disabled = true;
    submitBtn.textContent = '识别中...';

    fetch('/predict', {
      method: 'POST',
      body: formData
    }).then(resp => resp.json())
      .then(data => {
        submitBtn.disabled = false;
        submitBtn.textContent = '开始识别';

        if (!data.success) {
          msgDiv.textContent = data.msg || '识别失败，请重试。';
          return;
        }

        previewsDiv.classList.remove('hidden');

        labelP.textContent = '判断结果：' + data.label;

        probDiv.innerHTML = '';
        const probs = data.probabilities || {};
        Object.keys(probs).forEach(key => {
          const p = document.createElement('p');
          const value = (probs[key] * 100).toFixed(2);
          p.textContent = key + '：' + value + '%';
          probDiv.appendChild(p);
        });

        resultDiv.classList.remove('hidden');
      })
      .catch(err => {
        console.error(err);
        submitBtn.disabled = false;
        submitBtn.textContent = '开始识别';
        msgDiv.textContent = '请求异常，请检查后端是否启动。';
      });
  });
});

