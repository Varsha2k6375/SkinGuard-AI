// =====================================
// SkinGuard AI
// Classification Page
// =====================================

// Continue to Explainable AI Page

function nextPage(){

    window.location.href="/gradcam";

}

// Go Back

function previousPage(){

    window.history.back();

}

// Show Loading Animation

function showLoading(){

    const btn=document.getElementById("predictBtn");

    btn.innerHTML="Classifying...";

    btn.disabled=true;

}
// =====================================
// Image Preview
// =====================================

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");

if (imageInput && previewImage) {

    imageInput.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) return;

        const reader = new FileReader();

        reader.onload = function (e) {

            previewImage.src = e.target.result;

        };

        reader.readAsDataURL(file);

    });

}

// =====================================
// Predict Disease
// =====================================

const predictButton = document.getElementById("predictBtn");

if (predictButton) {

    predictButton.addEventListener("click", function () {

        const file = imageInput.files[0];

        if (!file) {

            alert("Please select an image.");

            return;

        }

        showLoading();

        const formData = new FormData();

        formData.append("image", file);

        fetch("/classification/predict", {

            method: "POST",

            body: formData

        })

        .then(response => response.json())

        .then(data => {

            predictButton.innerHTML = "Predict Disease";
            predictButton.disabled = false;

            if (data.success) {

                document.getElementById("resultCard").style.display = "block";

                document.getElementById("diseaseName").innerHTML = data.disease;

                document.getElementById("confidenceScore").innerHTML =
                    data.confidence + "%";

            }

            else {

                alert(data.message);

            }

        })

        .catch(error => {

            predictButton.innerHTML = "Predict Disease";
            predictButton.disabled = false;

            console.error(error);

            alert("Prediction Failed.");

        });

    });

}