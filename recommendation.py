from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for
)


# ==========================================
# Recommendation Blueprint
# ==========================================

recommendation = Blueprint(
    "recommendation",
    __name__
)


# ==========================================
# Recommendation Page
# ==========================================

@recommendation.route("/recommendation")
def recommendation_page():

    # --------------------------------------
    # Check Classification
    # --------------------------------------

    if "classification_image" not in session:

        return redirect(
            url_for(
                "classification.classification_page"
            )
        )


    # --------------------------------------
    # Get Results
    # --------------------------------------

    disease = session.get(
        "disease",
        "Unknown"
    )

    confidence = session.get(
        "confidence",
        0
    )

    severity = session.get(
        "severity",
        "Unknown"
    )


    # --------------------------------------
    # Recommendation
    # --------------------------------------

    if severity == "High":

        recommendation_text = (
            "High severity has been detected. "
            "The user is advised to consult a "
            "qualified dermatologist for further "
            "clinical evaluation."
        )

    elif severity == "Medium":

        recommendation_text = (
            "Medium severity has been detected. "
            "Regular monitoring and professional "
            "medical consultation are recommended."
        )

    else:

        recommendation_text = (
            "Low severity has been detected. "
            "Continue regular skin monitoring and "
            "consult a dermatologist if any "
            "unusual changes are observed."
        )


    # --------------------------------------
    # Save Recommendation
    # --------------------------------------

    session[
        "recommendation"
    ] = recommendation_text


    # --------------------------------------
    # Display Page
    # --------------------------------------

    return render_template(

        "recommendation.html",

        disease=disease,

        confidence=confidence,

        severity=severity,

        recommendation=recommendation_text,

        algorithm=(
            "Rule-Based Clinical Recommendation"
        ),

        status=(
            "Personalized Recommendation Generated"
        )

    )