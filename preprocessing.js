// ==========================================
// SkinGuard AI - Preprocessing Page
// preprocessing.js
// ==========================================

document.addEventListener("DOMContentLoaded", function () {

    console.log("Preprocessing Page Loaded");

    // Animate Cards
    const cards = document.querySelectorAll(".card");

    cards.forEach((card, index) => {

        card.style.opacity = "0";
        card.style.transform = "translateY(30px)";

        setTimeout(() => {

            card.style.transition = "all 0.6s ease";
            card.style.opacity = "1";
            card.style.transform = "translateY(0px)";

        }, index * 300);

    });

    // Animate Process List
    const processItems = document.querySelectorAll(".process-list li");

    processItems.forEach((item, index) => {

        item.style.opacity = "0";

        setTimeout(() => {

            item.style.transition = "0.5s";
            item.style.opacity = "1";

        }, 1200 + (index * 200));

    });

});


// ==========================================
// Continue Button
// ==========================================

function goToSegmentation() {

    window.location.href = "/segmentation";

}


// ==========================================
// Image Zoom Effect
// ==========================================

const images = document.querySelectorAll(".card img");

images.forEach(image => {

    image.addEventListener("mouseenter", () => {

        image.style.transform = "scale(1.05)";
        image.style.transition = "0.4s";

    });

    image.addEventListener("mouseleave", () => {

        image.style.transform = "scale(1)";

    });

});


// ==========================================
// Loading Animation (Optional)
// ==========================================

function showLoading() {

    const loader = document.getElementById("loader");

    if (loader) {

        loader.style.display = "flex";

    }

}


// ==========================================
// Hide Loading
// ==========================================

function hideLoading() {

    const loader = document.getElementById("loader");

    if (loader) {

        loader.style.display = "none";

    }

}


// ==========================================
// Display Current Time (Optional)
// ==========================================

function updateTime() {

    const now = new Date();

    const timeElement = document.getElementById("currentTime");

    if (timeElement) {

        timeElement.innerHTML = now.toLocaleTimeString();

    }

}

setInterval(updateTime, 1000);