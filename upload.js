// ==========================
// IMAGE PREVIEW
// ==========================

const imageInput = document.getElementById("image");

const preview = document.getElementById("previewImage");

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) return;

    const allowed = [

        "image/png",
        "image/jpeg",
        "image/jpg"

    ];

    if (!allowed.includes(file.type)) {

        alert("Only JPG, JPEG and PNG images are allowed.");

        imageInput.value = "";

        return;

    }

    const reader = new FileReader();

    reader.onload = function (event) {

        preview.src = event.target.result;

    }

    reader.readAsDataURL(file);

});


// ==========================
// CHECK IMAGE
// ==========================

function checkImageSelected() {

    if (preview.src == "") {

        alert("Please Capture or Upload an Image.");

        return false;

    }

    return true;

}