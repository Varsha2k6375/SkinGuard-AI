from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)

from werkzeug.utils import secure_filename

import os


# ============================================================
# HOME BLUEPRINT
# ============================================================

home = Blueprint(
    "home",
    __name__
)


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "static/uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


# ============================================================
# CREATE UPLOAD FOLDER
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK ALLOWED FILE
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME PAGE
# ============================================================

@home.route("/")
def index():

    return render_template(
        "home.html"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

@home.route(
    "/upload",
    methods=["POST"]
)
def upload():

    # --------------------------------------------------------
    # Check whether image exists
    # --------------------------------------------------------

    if "image" not in request.files:

        flash(
            "Please select an image."
        )

        return redirect(
            url_for("home.index")
        )


    # --------------------------------------------------------
    # Get uploaded file
    # --------------------------------------------------------

    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        flash(
            "Please select an image."
        )

        return redirect(
            url_for("home.index")
        )


    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        flash(
            "Only JPG, JPEG and PNG images are allowed."
        )

        return redirect(
            url_for("home.index")
        )


    # ========================================================
    # SECURE FILE NAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    # ========================================================
    # PREVENT OLD IMAGE REUSE
    # ========================================================

    # If another image with the same filename already exists,
    # create a new unique filename.

    base_name, extension = os.path.splitext(
        filename
    )


    counter = 1

    original_filename = filename


    while os.path.exists(
        os.path.join(
            UPLOAD_FOLDER,
            filename
        )
    ):

        filename = (
            f"{base_name}_{counter}{extension}"
        )

        counter += 1


    # ========================================================
    # COMPLETE IMAGE PATH
    # ========================================================

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    file.save(
        filepath
    )


    # ========================================================
    # SAVE CURRENT IMAGE IN SESSION
    # ========================================================

    session["uploaded_image"] = filepath.replace(
        "\\",
        "/"
    )


    # --------------------------------------------------------
    # Reset previous processing results
    # --------------------------------------------------------

    session.pop(
        "quality_completed",
        None
    )

    session.pop(
        "classification_image",
        None
    )

    session.pop(
        "disease",
        None
    )

    session.pop(
        "confidence",
        None
    )

    session.pop(
        "class_index",
        None
    )

    session.pop(
        "gradcam_completed",
        None
    )

    session.pop(
        "severity",
        None
    )


    # ========================================================
    # DEBUG
    # ========================================================

    print()
    print("=" * 60)
    print("IMAGE UPLOAD COMPLETED")
    print("=" * 60)

    print(
        "Original File :",
        original_filename
    )

    print(
        "Saved File    :",
        filename
    )

    print(
        "Image Path    :",
        session["uploaded_image"]
    )

    print(
        "Session Saved : YES"
    )

    print("=" * 60)
    print()


    # ========================================================
    # GO TO IMAGE QUALITY PAGE
    # ========================================================

    return redirect(
        url_for(
            "image_quality.quality"
        )
    )