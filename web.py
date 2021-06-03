# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
import os
import librosa
from hparams import create_hparams
from model import Model
from soundfile import write


app = Flask("app", template_folder="templates", static_folder="static")
@app.route('/', methods=['POST', 'GET'])

def upload():

    if request.method == 'POST':
        solver(request)
        return redirect(url_for('upload'))
    return render_template('web.html')

def solver(request):

    path_dir = 'static/audio'
    os.makedirs(path_dir, exist_ok=True)
    hparams = create_hparams()
    # model_name = request.form.get('model')
    f = request.files['file']
    y, sr = librosa.load(f, sr=hparams.SR)

    model = Model(hparams)
    checkpoint = "/Users/jlqian/Desktop/FD-LAMT/qian/G-MSS-main-2/trained_models/unet2-1/best.pkl"
    insts_wav = model.separate(y.reshape(1, y.shape[0]), hparams, checkpoint)

    insts_name = ['drums', 'bass', 'other', 'vocals', 'accompaniment']
    for idx, inst_name in enumerate(insts_name):
        write(os.path.join(path_dir, inst_name+'.wav'), insts_wav[idx], hparams.SR)


if __name__ == '__main__':

    app.run(debug=True)

