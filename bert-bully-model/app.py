from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)
socketio = SocketIO(app)

model_path = os.path.abspath(".")
print("📦 Loading BERT model from:", model_path)
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()
print("✅ BERT model loaded")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Bert_Chat</title>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
    <style>
        :root {
            --primary: #B5828C;
            --accent1: #FFCDB2;
            --accent2: #FFB4A2;
            --accent3: #E5989B;
            --background: #FFF5F3;
        }
        body {
            margin: 0;
            height: 100vh;
            font-family: 'Roboto', sans-serif;
            color: var(--primary);
            background: linear-gradient(270deg, #FFCDB2, #FFB4A2, #E5989B, #B5828C, #FFCDB2);
            background-size: 400% 100%;
            animation: gradientShift 18s ease-in-out infinite;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        /* Animated bubbles */
        .bubbles {
            position: fixed;
            top: 0; left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }
        .bubbles span {
            position: absolute;
            bottom: -100px;
            width: 20px;
            height: 20px;
            background: rgba(229, 152, 155, 0.18); /* #E5989B, more transparent */
            border-radius: 50%;
            box-shadow:
                0 0 10px rgba(255, 205, 178, 0.22),
                0 0 20px rgba(255, 180, 162, 0.16);
            animation: bubbleUp linear infinite;
            animation-duration: calc(22s / var(--i));
            left: calc(10% * var(--i));
            filter: drop-shadow(0 0 4px rgba(181, 130, 140, 0.22));
            opacity: 0.6;
        }
        @keyframes bubbleUp {
            0% {
                transform: translateY(100vh) scale(0.5);
                opacity: 0;
            }
            50% {
                opacity: 0.7;
            }
            100% {
                transform: translateY(-10vh) scale(1.2);
                opacity: 0;
            }
        }
        #login {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            gap: 20px;
            background: rgba(255, 255, 255, 0.85);
            z-index: 1;
            position: relative;
            color: var(--primary);
        }
        #login input, #login button {
            padding: 15px 25px;
            border-radius: 30px;
            border: 2px solid var(--accent3);
            background: var(--accent1);
            color: var(--primary);
            font-size: 16px;
            transition: all 0.3s ease;
        }
        #login input:focus {
            outline: none;
            border-color: var(--accent2);
        }
        #login button {
            background: var(--accent3);
            color: var(--accent1);
            font-weight: bold;
            cursor: pointer;
        }
        #login button:hover {
            background: var(--accent2);
            color: var(--primary);
        }
        #chat-container {
            display: none;
            flex: 1;
            flex-direction: column;
            height: 100vh;
            z-index: 1;
            position: relative;
            background: rgba(255, 255, 255, 0.85);
            color: var(--primary);
        }
        #chat {
            flex: 1;
            overflow-y: auto;
            padding: 24px 12px 12px 12px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            background: transparent;
        }
        .message {
            max-width: 70%;
            padding: 14px 20px;
            border-radius: 20px;
            background: var(--accent1);
            color: var(--primary);
            border: 1.5px solid var(--accent3);
            box-shadow: 0 2px 8px rgba(181, 130, 140, 0.3);
            word-break: break-word;
            animation: fadeIn 0.2s;
            position: relative;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px);}
            to { opacity: 1; transform: translateY(0);}
        }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--accent2), var(--accent3));
            color: #fff;
            border-top-right-radius: 0;
        }
        .message.other {
            align-self: flex-start;
            background: var(--accent1);
            color: var(--primary);
            border-top-left-radius: 0;
        }
        .message.bully {
            border: 2.5px solid #ff6f61; /* soft red for bully highlight */
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { border-color: #ff6f61; }
            50% { border-color: #ff9a8d; }
            100% { border-color: #ff6f61; }
        }
        .user-info {
            font-size: 0.92em;
            font-weight: bold;
            margin-bottom: 2px;
            color: var(--accent3);
        }
        .message.user .user-info {
            color: var(--accent1);
        }
        .bully-warning {
            color: #ff6f61;
            font-weight: bold;
            margin-left: 10px;
            font-size: 1em;
        }
        #input-container {
            padding: 18px 12px;
            background: var(--accent1);
            border-top: 1.5px solid var(--accent3);
        }
        #form {
            display: flex;
            gap: 10px;
        }
        #msg {
            flex: 1;
            padding: 14px 18px;
            border-radius: 28px;
            border: 2px solid var(--accent3);
            background: var(--background);
            color: var(--primary);
            font-size: 16px;
        }
        #msg:focus {
            outline: none;
            border-color: var(--accent2);
        }
        button {
            background: var(--accent3);
            color: var(--accent1);
            border: none;
            border-radius: 28px;
            padding: 0 22px;
            font-weight: bold;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.2s;
        }
        button:hover {
            background: var(--accent2);
            color: var(--primary);
        }
        @media (max-width: 600px) {
            #chat { padding: 10px 4px 4px 4px;}
            .message { max-width: 92%; font-size: 0.97em;}
            #input-container { padding: 10px 4px;}
        }
    </style>
</head>
<body>
    <div class="bubbles">
      <span style="--i:11;"></span>
      <span style="--i:18;"></span>
      <span style="--i:14;"></span>
      <span style="--i:23;"></span>
      <span style="--i:16;"></span>
      <span style="--i:19;"></span>
      <span style="--i:12;"></span>
      <span style="--i:20;"></span>
      <span style="--i:15;"></span>
      <span style="--i:13;"></span>
    </div>

    <div id="login">
        <h2>Welcome to Cyber Chat 💬</h2>
        <input type="text" id="username" placeholder="Enter your name..." maxlength="18" />
        <button onclick="connectUser()">Join Chat</button>
    </div>

    <div id="chat-container">
        <div id="chat"></div>
        <div id="input-container">
            <form id="form" autocomplete="off">
                <input id="msg" placeholder="Type your message..." autocomplete="off" />
                <button type="submit">Send ➤</button>
            </form>
        </div>
    </div>

    <script>
        var socket = io();
        var currentUser = '';
        function connectUser() {
            const username = document.getElementById('username').value.trim();
            if (username) {
                currentUser = username;
                document.getElementById('login').style.display = 'none';
                document.getElementById('chat-container').style.display = 'flex';
                document.getElementById('msg').focus();
            }
        }
        document.getElementById('form').addEventListener('submit', function(e) {
            e.preventDefault();
            const input = document.getElementById('msg');
            if (input.value.trim()) {
                socket.emit('message', {
                    user: currentUser,
                    text: input.value
                });
                input.value = '';
            }
        });
        socket.on('message', function(data) {
            const chat = document.getElementById('chat');
            const isCurrentUser = data.user === currentUser;
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isCurrentUser ? 'user' : 'other'}${data.is_bully ? ' bully' : ''}`;
            messageDiv.innerHTML = `
                <div class="user-info">${data.user}</div>
                ${data.text}
                ${data.is_bully ? '<span class="bully-warning">🚨 Bully Detected</span>' : ''}
            `;
            chat.appendChild(messageDiv);
            chat.scrollTop = chat.scrollHeight;
        });
        document.getElementById('username').addEventListener('keyup', function(e) {
            if (e.key === 'Enter') connectUser();
        });
    </script>
</body>
</html>

"""



@app.route('/')
def index():
    return render_template_string(html)

@socketio.on('message')
def handle_message(data):
    user = data['user']
    text = data['text']
    print(f"🗨️ {user}: {text}")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    is_bully = bool(prediction)
    print(f"🔍 Prediction: {prediction} {'(BULLY)' if is_bully else '(OK)'}")

    emit("message", {"user": user, "text": text, "is_bully": is_bully}, broadcast=True)

if __name__ == '__main__':
    print("🚀 Launching Chat App...")
    socketio.run(app, host='0.0.0.0', port=5000)