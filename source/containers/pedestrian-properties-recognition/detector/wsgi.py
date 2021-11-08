import predictor as ip_camera_ai_cognition_app

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = ip_camera_ai_cognition_app.app
