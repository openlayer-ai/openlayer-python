from .sklearn import add
from .transformers import add
from .tensorflow import add
from .pytorch import add
import subprocess
import webbrowser
from threading import Timer


def open_browser():
    webbrowser.open_new('http://localhost:5000/')


def commit(model_name):
    Timer(15, open_browser).start()
    subprocess.call(["bentoml", "serve", model_name+":latest"])


def push(version="v1"):
    subprocess.call(["bentoml", "lambda", "deploy", "lambda-deploy", "-b", "SklearnTextTemplateModel:"+version])