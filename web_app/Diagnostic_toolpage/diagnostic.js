function upload() {
    const fileUploadInput = document.querySelector(".file-uploader");
    const diseaseStatus = document.getElementById("disease-status");
    const recommendationText = document.getElementById("recommendation-text");

    if (!fileUploadInput.value) {
      return;
    }

    const image = fileUploadInput.files[0];

    if (!image.type.includes("image")) {
      return alert("Only images are allowed!");
    }

    if (image.size > 10_000_000) {
      return alert("Maximum upload size is 10MB!");
    }

    const fileReader = new FileReader();
    fileReader.readAsDataURL(image);

    fileReader.onload = (fileReaderEvent) => {
      const profilePicture = document.querySelector(".profile-picture");
      profilePicture.style.backgroundImage = `url(${fileReaderEvent.target.result})`;

      // Simulate a disease detection and recommendation process
      diseaseStatus.textContent = "Disease Detected: Example Disease";
      recommendationText.textContent = "Recommended Action: Apply XYZ pesticide and monitor the crop daily.";
    };

    // Here you can add your code to send the image to your server for actual disease detection
  }