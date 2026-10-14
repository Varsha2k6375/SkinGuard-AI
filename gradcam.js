// =====================================
// SkinGuard AI
// Grad-CAM Page
// =====================================

// Continue to Severity Prediction

function goToSeverity(){

    window.location.href="/severity";

}

// Go Back

function previousPage(){

    window.history.back();

}

// Loading Animation

function showLoading(){

    const button=document.querySelector("button");

    button.innerHTML="Generating Severity...";

    button.disabled=true;

}