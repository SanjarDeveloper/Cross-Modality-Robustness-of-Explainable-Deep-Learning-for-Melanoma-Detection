// Plain JavaScript: select an image, send it to Python, and display the result.
const $ = (id) => document.getElementById(id);
let busy = false;
let previewUrl = null;

$('image-file').onchange = () => {
  const file = $('image-file').files[0];
  // Release the previous preview from browser memory.
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  $('input-preview').hidden = !file;
  if (previewUrl) $('input-preview').src = previewUrl;
  else $('input-preview').removeAttribute('src');
  $('predict').disabled = !file || busy;
  $('prediction').hidden = true;
  $('empty').hidden = false;
  $('progress').textContent = '';
};

$('predict').onclick = async () => {
  const file = $('image-file').files[0];
  if (!file || busy) return;
  if (file.size > 10 * 1024 * 1024) {
    $('progress').textContent = 'Please choose an image no larger than 10 MB.';
    return;
  }
  busy = true;
  $('predict').disabled = true;
  $('image-file').disabled = true;
  $('prediction').hidden = true;
  $('empty').hidden = false;
  $('progress').textContent = 'Running model...';
  try {
    // Upload raw bytes; the server does not save the image to disk.
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file
    });
    const result = await response.json().catch(() => {
      throw new Error('The server returned an invalid response. Please try again.');
    });
    if (!response.ok) throw new Error(result.error || 'Unable to process this image.');
    $('score').textContent = result.score.toFixed(4);
    $('threshold').textContent = result.threshold.toFixed(4);
    $('decision').textContent = result.positive ? 'Prediction: melanoma' : 'Prediction: non-melanoma';
    $('overlay').hidden = !result.cam_available;
    if (result.overlay) $('overlay').src = result.overlay;
    else $('overlay').removeAttribute('src');
    $('cam-caption').textContent = result.cam_available
      ? 'Grad-CAM: regions contributing to the melanoma score.'
      : 'No informative Grad-CAM map is available.';
    $('empty').hidden = true;
    $('prediction').hidden = false;
    $('progress').textContent = '';
  } catch (error) {
    $('progress').textContent = `Error: ${error instanceof TypeError ? 'Cannot reach the server. Make sure the demo is running and try again.' : error.message}`;
  } finally {
    busy = false;
    $('predict').disabled = false;
    $('image-file').disabled = false;
  }
};
