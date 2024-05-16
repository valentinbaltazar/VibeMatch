from flask import Flask

def create_app():
    app = Flask(__name__, static_url_path='/static')
    # app.config['API_KEY'] = 'this can be any47278 thing'

    from .routes import routes

    app.register_blueprint(routes,url_prefix='/')

    return app
