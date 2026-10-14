import os

from flask import (
    Blueprint,
    render_template,
    jsonify,
    session,
    redirect,
    url_for
)

from classification.predict import predict


classification = Blueprint(
    "classification",
    __name__
)


def normalize_image_path(path):

    if not path:
        return None

    path = path.replace(
        "\\",
        "/"
    )

    path = path.lstrip("/")

    return path


@classification.route("/classification")
def classification_page():

    image_path = session.get(
        "uploaded_image"
    )

    if not image_path:

        return redirect(
            url_for("home.index")
        )

    image_path = normalize_image_path(
        image_path
    )

    if not os.path.isfile(
        image_path
    ):

        return redirect(
            url_for("home.index")
        )

    return render_template(

        "classification.html",

        image_path="/" + image_path,

        disease=session.get(
            "disease",
            "Unknown"
        ),

        confidence=session.get(
            "confidence",
            0
        ),

        class_index=session.get(
            "class_index",
            None
        )

    )


@classification.route(
    "/classification/predict",
    methods=["POST"]
)
def classify_image():

    image_path = session.get(
        "uploaded_image"
    )

    if not image_path:

        return jsonify({

            "success": False,

            "message":
                "No uploaded image found."

        })

    image_path = normalize_image_path(
        image_path
    )

    if not os.path.isfile(
        image_path
    ):

        return jsonify({

            "success": False,

            "message":
                "Uploaded image could not be found."

        })

    # -----------------------------------
    # Prediction
    # -----------------------------------

    try:

        result = predict(
            image_path
        )

    except Exception as e:

        print(
            "Classification Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "message":
                str(e)

        })

    # -----------------------------------
    # Required results
    # -----------------------------------

    disease = result.get(
        "disease",
        "Unknown"
    )

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    class_index = result.get(
        "class_index",
        0
    )

    probabilities = result.get(
        "probabilities",
        {}
    )

    # -----------------------------------
    # SAVE SESSION
    # -----------------------------------

    session["uploaded_image"] = image_path

    session["classification_image"] = image_path

    session["image_path"] = image_path

    session["disease"] = disease

    session["confidence"] = confidence

    session["class_index"] = class_index

    session["probabilities"] = probabilities

    # -----------------------------------
    # Debug
    # -----------------------------------

    print("\n===================================")
    print("CLASSIFICATION COMPLETED")
    print("===================================")

    print(
        "Image:",
        image_path
    )

    print(
        "Disease:",
        disease
    )

    print(
        "Confidence:",
        confidence
    )

    print(
        "Class Index:",
        class_index
    )

    print(
        "Probabilities:",
        probabilities
    )

    print("===================================\n")

    return jsonify({

        "success": True,

        "image_path":
            "/" + image_path,

        "disease":
            disease,

        "confidence":
            confidence,

        "class_index":
            class_index,

        "probabilities":
            probabilities

    })