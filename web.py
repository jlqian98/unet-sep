# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
import os
import librosa
from hparams import create_hparams
from model import Model
from soundfile import write
from pydub import AudioSegment


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
    if f.filename.split('.')[-1] == 'mp3':
        # f.save('static/audio/tmp.mp3', f)
        sound = AudioSegment.from_mp3(f)
        sound.export('static/audio/mixture.wav', format='wav')
        f = 'static/audio/mixture.wav'
    y, sr = librosa.load(f, sr=hparams.SR)
    
    mask = request.form.get('mask')
    hparams.hard_mask = (mask == 'hard')

    model_name = request.form.get('model')
    model = Model(hparams, model_name)
    if model_name == 'mmdensenet':
        checkpoint = "checkpoints/mmdensenet/best.pkl"
    elif model_name == 'unet':
        checkpoint = '../models/unet.pkl'
    insts_wav = model.separate(y.reshape(1, y.shape[0]), hparams, checkpoint)

    insts_name = hparams.insts
    for idx, inst_name in enumerate(insts_name):
        write(os.path.join(path_dir, inst_name+'.wav'), insts_wav[idx], hparams.SR)


if __name__ == '__main__':

    app.run(host='0.0.0.0')

