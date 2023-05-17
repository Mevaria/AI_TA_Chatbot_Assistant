from flask import Flask, render_template, request
import openai
import aiapi
from aiapi import chatbot, chatbot2

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['POST', 'GET'])
def index():

    return render_template('index.html', **locals())


@app.route('/chat')
def chat():
    input_text = request.args.get('input_text')
    output_text = chatbot(input_text)
    return output_text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
