from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()

@app.route('/', methods=["POST", "GET"])
def world():
    if request.method == "POST":
        print(request.form)
        dictt=request.form;
        fever = int(dictt['fever'])
        age = int(dictt['age'])
        pain = int(dictt['bodypain'])
        runny = int(dictt['runnynose'])
        diffbe = int(dictt['diffbreath']);
    inf = clf.predict_proba([[98,0,30,0,0]]);
    infProb = clf.predict_proba([[98, 0, 30, 0, 0]])[0][1];
    res = "hello"+str(infProb);
    return render_template('index.html')
    #return res;

if __name__=="__main__":
    app.run(debug=True)

