document.getElementById('imageUpload').addEventListener('change', function(event) {
    const reader = new FileReader();
    reader.onload = function() {
        document.getElementById('inputImage').src = reader.result;
    }
    reader.readAsDataURL(event.target.files[0]);
});

document.getElementById('checkPixelation').addEventListener('click', async function() {
    const fileInput = document.getElementById('imageUpload');
    if (fileInput.files.length === 0) {
        alert("Please upload an image first.");
        return;
    }
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    const response = await fetch('/check_pixelation', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('resultText').textContent = result.message;
});

document.getElementById('correctImage').addEventListener('click', async function() {
    const fileInput = document.getElementById('imageUpload');
    if (fileInput.files.length === 0) {
        alert("Please upload an image first.");
        return;
    }
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    const response = await fetch('/correct_image', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('outputImage').src = `data:image/jpeg;base64,${result.corrected_image}`;
    document.getElementById('psnrValue').textContent = `PSNR: ${result.psnr}`;
    document.getElementById('ssimValue').textContent = `SSIM: ${result.ssim}`;
});
