import mysql.connector

from flask import (
    Blueprint,
    render_template
)


# ==========================================
# Doctor Dashboard Blueprint
# ==========================================

doctor = Blueprint(
    "doctor",
    __name__
)


# ==========================================
# Database Connection
# ==========================================

def get_db_connection():

    connection = mysql.connector.connect(

        host="localhost",

        user="root",

        password="",

        database="skinguard_ai"

    )

    return connection


# ==========================================
# Doctor Dashboard
# ==========================================

@doctor.route("/doctor-dashboard")
def doctor_dashboard():

    connection = get_db_connection()

    cursor = connection.cursor(
        dictionary=True
    )

    cursor.execute("""

        SELECT
            id,
            patient_name,
            disease,
            confidence,
            severity,
            recommendation,
            image_path,
            analysis_date

        FROM patient_history

        ORDER BY analysis_date DESC

    """)

    patients = cursor.fetchall()

    cursor.close()

    connection.close()


    return render_template(

        "doctor_dashboard.html",

        patients=patients

    )