import os

from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for
)

from utils.preprocessing_utils import preprocess_image


# ============================================================
# PREPROCESSING BLUEPRINT
# ============================================================

preprocessing = Blueprint(
    "preprocessing",
    __name__
)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "static/uploads"
PREPROCESS_FOLDER = "static/preprocessed"

os.makedirs(
    PREPROCESS_FOLDER,
    exist_ok=True
)


# ============================================================
# PATH NORMALIZATION
# ============================================================

def normalize_path(path):

    if not path:
        return None

    return path.replace("\\", "/")


# ============================================================
# PREPROCESSING PAGE
# ============================================================

@preprocessing.route("/preprocessing")
def preprocessing_page():

    print("\n===================================")
    print("IMAGE PREPROCESSING - BM3D")
    print("===================================")

    # --------------------------------------------------------
    # GET SAME UPLOADED IMAGE
    # --------------------------------------------------------

    uploaded_image = session.get(
        "uploaded_image"
    )

    if not uploaded_image:

        return redirect(
            url_for("home.index")
        )

    uploaded_image = normalize_path(
        uploaded_image
    )

    # Remove leading slash
    uploaded_image = uploaded_image.lstrip("/")

    # --------------------------------------------------------
    # CHECK ORIGINAL IMAGE
    # --------------------------------------------------------

    if not os.path.isfile(
        uploaded_image
    ):

        print(
            "Uploaded image not found:",
            uploaded_image
        )

        return "Uploaded image not found"

    # --------------------------------------------------------
    # RUN BM3D PREPROCESSING
    # --------------------------------------------------------

    try:

        output_path = preprocess_image(
            uploaded_image,
            PREPROCESS_FOLDER
        )

    except Exception as e:

        print(
            "BM3D PREPROCESSING ERROR:",
            str(e)
        )

        return (
            "BM3D Preprocessing Failed: "
            + str(e)
        )

    # --------------------------------------------------------
    # CHECK OUTPUT
    # --------------------------------------------------------

    if not output_path:

        return (
            "BM3D preprocessed image "
            "was not generated."
        )

    output_path = normalize_path(
        output_path
    )

    output_path = output_path.lstrip("/")

    if not os.path.isfile(
        output_path
    ):

        print(
            "Preprocessed file missing:",
            output_path
        )

        return (
            "BM3D preprocessed image "
            "file not found."
        )

    # --------------------------------------------------------
    # SAVE SESSION
    # --------------------------------------------------------

    session["preprocessed_image"] = output_path

    session["preprocessing_completed"] = True

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "Original Image:",
        uploaded_image
    )

    print(
        "BM3D Preprocessed Image:",
        output_path
    )

    print(
        "BM3D preprocessing completed successfully."
    )

    # --------------------------------------------------------
    # BROWSER URLs
    # --------------------------------------------------------

    original_url = "/" + uploaded_image

    processed_url = "/" + output_path

    # --------------------------------------------------------
    # RENDER PAGE
    # --------------------------------------------------------

    return render_template(

        "preprocessing.html",

        original_image=original_url,

        processed_image=processed_url

    )