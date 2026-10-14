// ==========================
// START CAMERA
// ==========================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const previewImage = document.getElementById("previewImage");

async function startCamera() {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({

            video: true,
            audio: false

        });

        video.srcObject = stream;

    }

    catch (error) {

        alert("Unable to access camera.");

        console.log(error);

    }

}

startCamera();


// ==========================
// CAPTURE IMAGE
// ==========================

captureBtn.addEventListener("click", function () {

    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL("image/png");

    previewImage.src = imageData;

    sessionStorage.setItem("capturedImage", imageData);

});