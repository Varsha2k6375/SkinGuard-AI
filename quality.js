// ================================
// Quality Page Animation
// ================================

window.onload = function () {

    const score = document.querySelector(".score-value");

    if(score){

        score.style.transform = "scale(1.2)";

        setTimeout(() => {

            score.style.transform = "scale(1)";

        },800);

    }

};


// ================================
// Continue Button
// ================================

function nextPage(){

    window.location.href="/preprocessing";

}