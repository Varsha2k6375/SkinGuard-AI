from flask import Flask, redirect, url_for


# ==========================================
# IMPORT ROUTES
# ==========================================

from routes.home import home
from routes.image_quality import image_quality
from routes.preprocessing import preprocessing
from routes.segmentation import segmentation
from routes.classification import classification
from routes.gradcam import gradcam
from routes.severity import severity
from routes.recommendation import recommendation
from routes.report import report
from routes.doctor import doctor
from routes.patient import patient


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# SECRET KEY
# ==========================================

app.secret_key = "skinguard_ai_secret_key_2026"


# ==========================================
# REGISTER BLUEPRINTS
# ==========================================

app.register_blueprint(home)

app.register_blueprint(image_quality)

app.register_blueprint(preprocessing)

app.register_blueprint(segmentation)

app.register_blueprint(classification)

app.register_blueprint(gradcam)

app.register_blueprint(severity)

app.register_blueprint(recommendation)

app.register_blueprint(report)

app.register_blueprint(doctor)

app.register_blueprint(patient)

# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def index():

    return redirect(
        url_for("home.home_page")
    )


# ==========================================
# 404 ERROR
# ==========================================

@app.errorhandler(404)
def page_not_found(error):

    return """
    <h1>404 - Page Not Found</h1>
    <p>The requested page does not exist.</p>
    """, 404


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )