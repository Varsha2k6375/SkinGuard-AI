from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for
)


severity = Blueprint(
    "severity",
    __name__
)


@severity.route("/severity")
def severity_page():

    # -----------------------------------
    # Classification check
    # -----------------------------------

    if "classification_image" not in session:

        return redirect(
            url_for(
                "classification.classification_page"
            )
        )

    # -----------------------------------
    # Classification data
    # -----------------------------------

    disease = session.get(
        "disease",
        "Unknown"
    )

    confidence = float(
        session.get(
            "confidence",
            0
        )
    )

    # -----------------------------------
    # Severity score
    #
    # Use classification confidence
    # as the rule-based severity score.
    # -----------------------------------

    severity_score = round(
        confidence,
        2
    )

    # -----------------------------------
    # Severity level
    # -----------------------------------

    if severity_score >= 80:

        severity_level = "High"

        recommendation = (
            "High severity indication detected. "
            "Professional dermatological evaluation "
            "is recommended."
        )

    elif severity_score >= 60:

        severity_level = "Medium"

        recommendation = (
            "Medium severity indication detected. "
            "Further monitoring and professional "
            "dermatological consultation are recommended."
        )

    else:

        severity_level = "Low"

        recommendation = (
            "Low severity indication detected. "
            "Continue regular skin monitoring and "
            "consult a dermatologist if any visible "
            "changes occur."
        )

    # -----------------------------------
    # SAVE EVERYTHING
    # -----------------------------------

    session["severity"] = severity_level

    session["severity_score"] = severity_score

    session["recommendation"] = recommendation

    session["severity_algorithm"] = (
        "Rule-Based Severity Prediction"
    )

    session["severity_completed"] = True

    # -----------------------------------
    # Debug
    # -----------------------------------

    print("\n===================================")
    print("SEVERITY PREDICTION")
    print("===================================")

    print(
        "Disease:",
        disease
    )

    print(
        "Confidence:",
        confidence
    )

    print(
        "Severity:",
        severity_level
    )

    print(
        "Severity Score:",
        severity_score
    )

    print("===================================\n")

    # -----------------------------------
    # Page
    # -----------------------------------

    return render_template(

        "severity.html",

        disease=disease,

        confidence=confidence,

        severity=severity_level,

        severity_score=severity_score,

        recommendation=recommendation,

        algorithm=(
            "Rule-Based Severity Prediction"
        ),

        status=(
            "Severity Prediction Completed"
        )

    )