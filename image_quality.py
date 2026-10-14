from flask import (
    Blueprint,
    render_template,
    session,
    redirect,
    url_for
)

import cv2
import numpy as np
import os


# ============================================================
# IMAGE QUALITY BLUEPRINT
# ============================================================

image_quality = Blueprint(
    "image_quality",
    __name__
)


# ============================================================
# IMAGE QUALITY PAGE
# ============================================================

@image_quality.route("/quality")
def quality():

    # --------------------------------------------------------
    # Get the CURRENT uploaded image from session
    # --------------------------------------------------------

    image_path = session.get("uploaded_image")


    # --------------------------------------------------------
    # If no image was uploaded
    # --------------------------------------------------------

    if not image_path:

        return redirect(
            url_for("home.index")
        )


    # --------------------------------------------------------
    # Convert Windows path to web-friendly path
    # --------------------------------------------------------

    image_path = image_path.replace(
        "\\",
        "/"
    )


    # --------------------------------------------------------
    # Check whether image file exists
    # --------------------------------------------------------

    if not os.path.isfile(image_path):

        return redirect(
            url_for("home.index")
        )


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )


    # --------------------------------------------------------
    # Check image reading
    # --------------------------------------------------------

    if image is None:

        return render_template(

            "quality.html",

            image_path="/" + image_path,

            resolution="Not Available",

            blur="Invalid",

            brightness="Invalid",

            contrast="Invalid",

            noise="Invalid",

            visibility="Invalid",

            score=0,

            status="Invalid Image",

            message=(
                "Unable to read the uploaded image. "
                "Please upload another image."
            )

        )


    # ========================================================
    # IMAGE CONVERSION
    # ========================================================

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    # ========================================================
    # RESOLUTION CHECK
    # ========================================================

    height, width = gray.shape

    resolution = f"{width} × {height}"


    # ========================================================
    # BLUR DETECTION
    # ========================================================

    blur_value = cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).var()


    if blur_value >= 150:

        blur = "Excellent"

        blur_score = 20


    elif blur_value >= 80:

        blur = "Good"

        blur_score = 15


    else:

        blur = "Poor"

        blur_score = 5


    # ========================================================
    # BRIGHTNESS CHECK
    # ========================================================

    brightness_value = float(
        np.mean(gray)
    )


    if 80 <= brightness_value <= 180:

        brightness = "Normal"

        brightness_score = 20


    elif brightness_value < 80:

        brightness = "Low"

        brightness_score = 10


    else:

        brightness = "High"

        brightness_score = 10


    # ========================================================
    # CONTRAST CHECK
    # ========================================================

    contrast_value = float(
        np.std(gray)
    )


    if contrast_value >= 45:

        contrast = "Good"

        contrast_score = 20


    else:

        contrast = "Low"

        contrast_score = 10


    # ========================================================
    # NOISE CHECK
    # ========================================================

    noise_value = cv2.Laplacian(

        gray,

        cv2.CV_64F

    ).std()


    if noise_value < 20:

        noise = "Low"

        noise_score = 20


    elif noise_value < 40:

        noise = "Medium"

        noise_score = 15


    else:

        noise = "High"

        noise_score = 5


    # ========================================================
    # LESION VISIBILITY
    # ========================================================

    # Basic visibility indicator.
    # Detailed lesion segmentation is handled
    # in the segmentation module.

    visibility = "Good"

    visibility_score = 20


    # ========================================================
    # FINAL QUALITY SCORE
    # ========================================================

    score = (

        blur_score
        + brightness_score
        + contrast_score
        + noise_score
        + visibility_score

    )


    # ========================================================
    # QUALITY STATUS
    # ========================================================

    if score >= 90:

        status = "Excellent Image"

        message = (
            "Image quality is excellent "
            "and suitable for further processing."
        )


    elif score >= 70:

        status = "Good Image"

        message = (
            "Image quality is good "
            "and the image can be processed."
        )


    else:

        status = "Poor Image"

        message = (
            "Image quality is poor. "
            "Please upload a clearer skin image."
        )


    # ========================================================
    # SAVE QUALITY INFORMATION IN SESSION
    # ========================================================

    session["quality_score"] = score

    session["quality_status"] = status

    session["quality_completed"] = True


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    print()
    print("=" * 60)
    print("IMAGE QUALITY CHECK COMPLETED")
    print("=" * 60)

    print(
        "Current Image :",
        image_path
    )

    print(
        "Resolution    :",
        resolution
    )

    print(
        "Blur          :",
        blur
    )

    print(
        "Brightness    :",
        brightness
    )

    print(
        "Contrast      :",
        contrast
    )

    print(
        "Noise         :",
        noise
    )

    print(
        "Visibility    :",
        visibility
    )

    print(
        "Quality Score :",
        score
    )

    print(
        "Status        :",
        status
    )

    print("=" * 60)
    print()


    # ========================================================
    # SEND DATA TO QUALITY HTML
    # ========================================================

    return render_template(

        "quality.html",

        image_path="/" + image_path,

        resolution=resolution,

        blur=blur,

        brightness=brightness,

        contrast=contrast,

        noise=noise,

        visibility=visibility,

        score=score,

        status=status,

        message=message

    )