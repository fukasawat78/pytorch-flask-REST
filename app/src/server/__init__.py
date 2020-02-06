from flask import Flask
from . import hoge

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config/default')

blueprints = [hoge]
for blueprint in blueprints:
    app.register_blueprint(blueprint.app)