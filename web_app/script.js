function predict() {
    const input = document.getElementById('imageInput').files[0];
    if (!input) {
        alert('Please upload an image first.');
        return;
    }

    // Preview the image
    const previewImage = document.getElementById('previewImage');
    const reader = new FileReader();
    reader.onload = function(event) {
        previewImage.src = event.target.result;
        previewImage.style.display = 'block';
    }
    reader.readAsDataURL(input);

    // Simulate a prediction (this is where you'd call your backend)
    setTimeout(() => {
        document.getElementById('result').innerText = 'Predicted: Healthy Leaf';
    }, 1000);
}
