# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import flask
from flask import Flask, render_template, request # type: ignore
import random
import flask_cors
import json
import onnx
import onnxruntime
import random
from random import randrange, randint
import math

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = flask.Flask(__name__, template_folder='templates')
flask_cors.CORS(app) 

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    r = random.randint(0,9)
    response = flask.jsonify({'play': r})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/home')
def home():
    return render_template('main.html')

model_name = 'model.onnx'
onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)

EP_list = ['CPUExecutionProvider']

ort_session = onnxruntime.InferenceSession(model_name, providers=EP_list)

@app.route('/sendhere', methods=['POST'])
def sendhere():
    x = request.get_json()
    # gamearray=json.loads(x['gamearray'])
    # # res = flask.jsonify({"playthis":gamearray[2]})
    # # res.headers.add('Access-Control-Allow-Origin', '*')
    # # return res
    # data = "asdf"
    # res = flask.jsonify({"game":data})
    # res.headers['Content-Type'] = 'application/json'
    # res.headers.add('Access-Control-Allow-Origin', '*')
    # return res

    
    board = x['gamearray']
    ort_inputs = {ort_session.get_inputs()[0].name: oneHotter(board)}
    output = ort_session.run(None, ort_inputs)

    move=argmax(output[0])

    status = checkAnyWin(board)
    data = {
        "move":move, # output move from the model
        "status":status # status of game 0 for continue 1 for player win 
                   # 2 for ai win and 3 for tie
    }

    resp = flask.make_response(flask.jsonify(data))
    resp.headers['Content-Type'] = 'application/json'

    h = resp.headers
    # prepare headers for CORS authentication
    h['Access-Control-Allow-Origin' ] = '*'
    h['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    h['Access-Control-Allow-Headers'] = 'X-Requested-With, Content-Type'

    resp.headers = h
    return resp


def oneHotter(game):
    onehot=[]
    for i in game:
        if(i==0):
            onehot.append(1)
            onehot.append(0)
            onehot.append(0)
        elif(i==1):
            onehot.append(0)
            onehot.append(1)
            onehot.append(0)
        elif(i==2):
            onehot.append(0)
            onehot.append(0)
            onehot.append(1)
    return onehot

def argmax(inp):
    maximum=float('-inf')
    index=0
    for i in range(len(inp)):
        if(inp[i]>maximum):
            maximum=inp[i]
            index=i     
    return index        


def checkVertical(positions):
    if(  (positions[6]==positions[3]==positions[0]==2)
      or (positions[7]==positions[4]==positions[1]==2)
      or (positions[8]==positions[5]==positions[2]==2)):
        return 2
    if(  (positions[6]==positions[3]==positions[0]==1)
      or (positions[7]==positions[4]==positions[1]==1)
      or (positions[8]==positions[5]==positions[2]==1)):
        return 1
    return 0

def checkHorizontal(positions):
    if(  (positions[6]==positions[7]==positions[8]==2)
      or (positions[3]==positions[4]==positions[5]==2)
      or (positions[0]==positions[1]==positions[2]==2)):
        return 2
    if(  (positions[6]==positions[7]==positions[8]==1)
      or (positions[3]==positions[4]==positions[5]==1)
      or (positions[0]==positions[1]==positions[2]==1)):
        return 1
    return 0

def checkDiagonal(positions):
    if(  (positions[6]==positions[4]==positions[2]==2)
      or (positions[0]==positions[4]==positions[8]==2)):
        return 2
    if(  (positions[6]==positions[4]==positions[2]==1)
      or (positions[0]==positions[4]==positions[8]==1)):
        return 1
    return 0

def checkAnyWin(board):
    if(checkVertical(board)==1 or checkHorizontal(board)==1 or checkDiagonal(board)==1 ):
#         print("Player 1 wins")
        return 1
    if(checkVertical(board)==2 or checkHorizontal(board)==2 or checkDiagonal(board)==2 ):
#         print("Player 2 wins")
        return 2
    return 0    

def play(board, remain, action, turn):
    if(len(remain)>0):
        if(action in remain):
            board[action]=turn
            remain.remove(action)            
        else:
            return -10
        
    return checkAnyWin(board)

def oneHot(board):
    return torch.tensor( [(F.one_hot(torch.tensor(board), num_classes=3) ).tolist()], dtype=torch.float32).flatten()

def makeOneHot(player , pos):
    x = math.floor(random.random()*9)
    a = [0.0]*9
    a[pos] = player
    return a



# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()  