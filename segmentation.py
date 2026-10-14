import os

from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for
)

from utils.segmentation_utils import segment_image


segmentation = Blueprint(
    "segmentation",
    __name__
)


SEGMENT_FOLDER = "static/segmented"

os.makedirs(
    SEGMENT_FOLDER,
    exist_ok=True
)


def normalize_path(path):

    if not path:
        return None

    return path.replace("\\", "/")


@segmentation.route("/segmentation")
def segmentation_page():

    print("\n===================================")
    print("U-NET SEGMENTATION")
    print("===================================")

    # -----------------------------------
    # Get preprocessed image from session
    # -----------------------------------

    latest_file = session.get(
        "preprocessed_image"
    )

    if not latest_file:

        return redirect(
            url_for(
                "preprocessing.preprocessing_page"
            )
        )

    latest_file = normalize_path(
        latest_file
    ).lstrip("/")

    print(
        "Preprocessed image:",
        latest_file
    )

    # -----------------------------------
    # Check file
    # -----------------------------------

    if not os.path.isfile(
        latest_file
    ):

        print(
            "Preprocessed image not found:",
            latest_file
        )

        return "Preprocessed image not found."

    # -----------------------------------
    # Run U-Net
    # -----------------------------------

    try:

        segmented_path = segment_image(

            latest_file,

            SEGMENT_FOLDER

        )

    except Exception as e:

        print("\nSEGMENTATION ERROR")
        print(
            type(e).__name__,
            str(e)
        )

        return (
            "Segmentation Failed: "
            + str(e)
        )

    # -----------------------------------
    # Validate output
    # -----------------------------------

    if not segmented_path:

        return "Segmentation output was not generated."

    segmented_path = normalize_path(
        segmented_path
    ).lstrip("/")

    print(
        "Segmented output:",
        segmented_path
    )

    if not os.path.isfile(
        segmented_path
    ):

        return "Segmentation output file not found."

    # -----------------------------------
    # SAVE SESSION
    # -----------------------------------

    session["segmentation_image"] = segmented_path

    session["segmentation_mask"] = segmented_path

    session["segmentation_completed"] = True

    # -----------------------------------
    # Lesion area
    #
    # If your segment_image() does not
    # return area, keep this text.
    # -----------------------------------

    session["lesion_area"] = "Detected"

    session["segmentation_status"] = (
        "Segmentation Completed Successfully"
    )

    print(
        "Session segmentation_image:",
        session.get("segmentation_image")
    )

    # -----------------------------------
    # Browser URLs
    # -----------------------------------

    original_url = "/" + latest_file

    segmented_url = "/" + segmented_path

    return render_template(

        "segmentation.html",

        original_image=original_url,

        segmented_image=segmented_url,

        algorithm="U-Net",

        status=(
            "Segmentation Completed Successfully"
        )

    )