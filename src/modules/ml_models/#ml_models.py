"""NE MARCHE PAS"""
import subprocess

subprocess.call(['python', "src\modules\ml_models\optimisation_script.py"])
subprocess.call(['python', "src\modules\ml_models\prediction_script.py"])
subprocess.call(['python', "src\modules\ml_models\importance_script.py"])