#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import sys
path2proj = "/Users/user/PycharmProjects/DeepLearningPy"
sys.path.append(path2proj)
from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime
import numpy as np
from lib.flask.Form1 import *
import pyjade
import jinja2
import json

import model.DataPreprocessing as mdp
import model.ModelFitting as mmf
import app.sentiment_analysis_app as saa

#reload(sys)
#sys.setdefaultencoding("utf-8")

# define application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'atala0628'
app.debug = True
app.jinja_env.add_extension('pyjade.ext.jinja.PyJadeExtension')

moment = Moment(app)
manager = Manager(app)
bootstrap = Bootstrap(app)
site_title = "funwithdata:Deep"



# Routings
@app.route('/', methods=['GET', 'POST'])
def index():
    name, password, like = None, None, None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        password = form.password.data
        like = form.like.data
        form.name.data = u''
        form.password.data = u''

    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "DEEP LEARNINGS BY SOMEONESGARDEN.",
        'subtitle': "welcome, %s !! " % name if name is not None else ""
    }
    return render_template('index.html', form=form,essentials=essentials)


@app.route('/sentimentanalysis', methods=['GET', 'POST'])
def sentimentanalysis():
    if request.method == 'POST':
        form = request.form
    else:
        form = request.args
    param = json.loads(form.get('param'))
    res = {}

    print(param)

    if param['action'] == 'analyse':
        review = param['review']

        label, proba = "", 0.0
        if review != "":
            label, proba = saa.classify(review)

        res['action'] = 'analyse'
        res['label'] = label
        res['proba'] = proba

    elif param['action'] == 'feedback':
        review = param['review']
        prediction = param['prediction']

        label = {0: 'negative', 1: 'positive'}
        inv_label = {'negative': 0, 'positive': 1}

        y = inv_label[prediction]
        y = int(not y)

        print("Wrong Result. NEED FEEDBACK. train data as :%s %d" % (review, y))

        saa.train(review, y)
        saa.sqlite_entry(saa.db, review, y)

        res['action'] = 'feedback'
        res['label'] = label[y]

    saa.show_review_db()
    return jsonify(res)


@app.route('/math')
def math():
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "MATH.",
        'subtitle': ""
    }
    return render_template('math_j.jade', essentials=essentials, test1=123)


@app.route('/math/<param>')
@app.route('/math/<param>/')
def math_page_param(param):
    title = param if param is not "" else "math."
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': title,
        'subtitle': "",
    }
    return render_template('math/math{0}.html'.format(int(param)), essentials=essentials)


@app.route('/latex')
def latex():
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "LATEX.",
        'subtitle': ""
    }
    return render_template('latex.html', essentials=essentials)


@app.route('/factory')
def factory():
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "FACTORY.",
        'subtitle': ""
    }
    return render_template('factory.html', essentials=essentials)


@app.route('/pipelines')
def pipelines():
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "PIPELINES.",
        'subtitle': ""
    }
    return render_template('pipelines.html', essentials=essentials)


@app.route('/ensembles')
def ensemples():
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "Ensembles.",
        'subtitle': ""
    }
    return render_template('ensembles.html', essentials=essentials)


@app.route('/d3')
def d3_page():
    title = "D3.js for data visualization."
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': title,
        'subtitle': "",
    }
    return render_template('d3.html', essentials=essentials)


@app.route('/d3/<param>')
@app.route('/d3/<param>/')
def d3_page_param(param):
    title = param if param is not "" else "D3.js for data visualization."
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': title,
        'subtitle': "",
    }
    return render_template('d3/case{0}.html'.format(int(param)), essentials=essentials)


@app.route('/cgi', methods=['GET', 'POST'])
def cgi():
    if request.method == 'POST':
        form = request.form
    else:
        form = request.args
    print(form)
    cgi_type = form.get('cgi_type')

    if cgi_type == 'linear_regression':
        eta = float(form.get('eta'))
        epoch = int(form.get('epoch'))
        print(eta)
        print(epoch)
        # linear_regression_main.main(learning_rate=eta, EPOCH=epoch)

    res = {
        'result': True,
        'cgi_type': cgi_type
    }
    return jsonify(res)


@app.route('/post', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        name = request.form['name']
        return render_template('index.html', essentials={'title': name, 'subtitle': 'asdfsdfasd'})
    else:
        return redirect(url_for('index'))
        # return redirect(url_for('static', filename='index'))


@app.route('/upload', methods=['POST'])
def upload():
    form = request.form
    param = json.loads(form.get('param'))
    status, message, url = False, '', ''

    if param['step'] == "step1":
        if param['action'] == "url":
            url = param['url']
            print("URL")
            status = True
            message = 'URL ACTION\n'

        if param['action'] == "upload":
            print("UPLOAD")
            files = request.files
            if len(files) > 0:
                the_file = files['file']
                the_file.save("./static/uploaded/" + the_file.filename)
                status = True
                message = 'Upload succeed\n'
                url = "/Users/user/PycharmProjects/DeepLearningPy/static/uploaded/"+the_file.filename
            else:
                status = False
                message = 'Upload Failed\n'

        # pre-process
        param['status'] = status
        param['message'] = message
        param['url'] = url
        preprocess = mdp.DataPreProcess()
        param = preprocess.validate_url(param=param)
        # print param

    elif param['step'] == "step2":

        mf = mmf.Fitting(param)
        mf.predict_with_param()

        param['status'] = True
        param['message'] = 'here is step2!!!'

        print("shape:{0}".format(mf.df.shape))

    return jsonify(param)


# ERROR HANDLINGS
class MyNotFoundException(Exception):
    pass


@app.route('/404')
@app.errorhandler(404)
@app.errorhandler(500)
@app.errorhandler(503)
@app.errorhandler(504)
@app.errorhandler(501)
def error_handler(err):
    # abort(404)
    return render_template('404.html', error=err), 404


@app.errorhandler(ValueError)
@app.errorhandler(UnicodeDecodeError)
@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def error_handler(err):
    # return 'UnicodeDecodeError', 404
    return render_template('404.html', error=err)


if __name__ == '__main__':
    app.debug = True
    app.run('127.0.0.1', 8001)
    # manager.run()
    # コマンドライン >>
    # python app.py runserver -p 8002 -h 127.0.0.1 -d
