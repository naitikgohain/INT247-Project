from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

file = open('model_neural.pkl', 'rb')

clf = pickle.load(file)
file.close()

@app.route('/', methods=["POST", "GET"])
def world():
    if request.method == "POST":
        print(request.form)
        dictt=request.form;
        fever = float(dictt['fever'])
        age = int(dictt['age'])
        pain = int(dictt['bodypain'])
        runny = int(dictt['runnynose'])
        diffbe = int(dictt['diffbreath']);
        params=[fever, age, pain, runny, diffbe];
        #if (params.ndim == 1):
        params = np.array([params])
        y_pred=clf.predict(params)[0][0]
        #print(y_pred)
        y_pred=int(round(y_pred*100))
        #print(y_pred)
        #print(y_pred)
        return render_template('result.html', inf=y_pred);
    
    #inf = clf.predict_proba([[98,0,30,0,0]]);
    #infProb = clf.predict_proba([[98, 0, 30, 0, 0]])[0][1];
    
    #res = "hello"+str(y_pred);
    return render_template('index.html')
    #return res;

if __name__=="__main__":
    app.run(debug=True)

