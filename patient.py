from flask import (
    Blueprint,
    render_template,
    request,
    session,
    redirect,
    url_for
)


patient = Blueprint(
    "patient",
    __name__
)


@patient.route(
    "/patient-details",
    methods=["GET", "POST"]
)
def patient_details():

    if request.method == "POST":

        session["patient_name"] = request.form.get(
            "patient_name"
        )

        session["patient_age"] = request.form.get(
            "age"
        )

        session["patient_gender"] = request.form.get(
            "gender"
        )

        return redirect(
            url_for(
                "classification.classification_page"
            )
        )

    return render_template(
        "patient_details.html"
    )