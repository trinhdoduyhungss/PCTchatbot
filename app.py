from flask import Flask
from flask import jsonify
from flask import request, render_template
from flask_cors import CORS
import requests
import pymongo
import json
import ast
import wolframalpha
from api.wolframalpha import search_and_solve
from pyvi import ViTokenizer
from vi_stop_words import STOP_WORDS

from gensim.models.word2vec import Word2Vec
from text_classifier import TextClassifier
import time, random

app = Flask(__name__, template_folder='template')
CORS(app)

word2vec_dict = Word2Vec.load('VNCorpus9.bin')
model_path = 'sentiment_model9.h5'
keras_text_classifier = TextClassifier(word2vec_dict=word2vec_dict, model_path=model_path,max_length=20, n_epochs=20, n_class=8)
labels = {1: 'chemistry', 2: 'general_asking', 6: 'math', 3: 'goodbye', 4: 'hello', 5: 'introduction', 7: 'thanks',
          0: 'ask_weather'}

server = pymongo.MongoClient("mongodb+srv://Hung:1@cluster0-3izqd.mongodb.net/test?retryWrites=true&w=majority")
database = server['ChatApp']

messageCollection = database['messages']


def seed_data():
    if messageCollection.count() > 0:
        print("Collection is not null")
        return
    else:
        content = """Lorem ipsum dolor sit amet consectetur adipisicing elit. Fugiat ab reiciendis unde debitis, veritatis quis delectus eaque, dolores quaerat, fuga veniam repudiandae consequatur dignissimos eos doloribus assumenda distinctio! Laboriosam, placeat."""
        for i in range(10):
            record = None
            if (i % 2 == 0):
                record = {
                    "isBot": False,
                    "content": content,
                    "time": time.time()
                }
            else:
                record = {
                    "isBot": True,
                    "content": content,
                    "time": time.time()
                }
            messageCollection.insert_one(record)
            time.sleep(2)
        print("Seeded successfully")
seed_data()

response = {
    "hello": [
        "Xin chào bạn.",
        "Chào bạn, tôi có thể giúp gì cho bạn",
        "Chào bạn, ngày mới tốt lành.",
        "Xin chào.",
        "Bạn là ai, mình không biết, bạn out ra đi :V",
        "Để mình yên, hỏi gì mà hỏi, google đâu mà đéo hỏi, làm phiền tui :V"
      ],
    "goodbye":[
        "Tạm biêt.",
        "Hẹn gặp lại",
        "Chào tạm biệt",
        "Hẹn gặp lại bạn lần sau",
        "Tạm biệt, hẹn gặp lại"
      ],
    "thanks": [
        "Hân hạnh",
        "Luôn sẵn lòng :)",
        "Cảm ơn bạn, giúp được bạn là vinh dự của tôi",
        "Cảm ơn bạn, Rất vui khi được giúp đỡ bạn"
      ],
    "introduction": [
        "Tôi là chat bot có khả năng giúp bạn giải những vấn đề cơ bản trong cuộc sống",
        "Tôi là chat bot, hi vọng có thể giúp bạn một ít",
        "Là một chatbot tôi có thể giúp bạn trong vài lĩnh vực như toán, hóa, lịch sử ...",
        "Tôi có thể giúp bạn vài bài tập đơn giản như cân bằng hóa học, giải phương trình, ...",
        "Tôi là chatbot, rất hân hạnh làm quen với bạn"
      ],
    "cant":[
        "Thế bạn nói xem vì sao mình phải trả lời bạn :) ?",
        "Rất tiếc bạn không có thứ mình cần và bạn hỏi không lịch sự nên mình không cần trả lời bạn :)))",
        "Xin lỗi, tôi chưa hiểu câu hỏi của bạn",
        "Xin lỗi, tôi chưa hiểu câu hỏi của bạn",
        "Xin lỗi, tôi chưa hiểu câu hỏi của bạn",
        "Tôi nghĩ vấn đề này ngoài khả năng của tôi.",
        "Xin lỗi bạn, tôi chưa được học về vấn đề này",
        "Tôi chưa thể giúp bạn ngay bây giờ, tôi sẽ cải thiện sau",
        "Xin lỗi, tôi chưa hiểu câu hỏi của bạn",
        "Xin lỗi, hiện tại tôi chưa thể giúp bạn",
        "Xin lỗi, hiện tại tôi chưa thể giúp bạn",
        "Vấn đề này khá mới với tôi, tôi có thể học sau, xin lỗi bạn",
      ],
    "general_asking": [
        "Đây là kết quả của tôi, bạn có thể tham khảo:\n",
        "Tôi đã có kết quả:\n"
      ],
    "calculator": [
        "Đây là kết quả của tôi, bạn có thể tham khảo:\n",
        "Đây là kết quả: \n",
        "Kết quả nè:\n",
        "Kết quả:\n"
      ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/messages', methods=['GET'])
def get_messages():
    res = messageCollection.find({}, {'_id': False}).sort("time", 1)
    return json.dumps({'results': list(res)})


@app.route('/api/v1/messages', methods=['POST'])
def save_message():
    try:
        # Create new users
        try:
            body = request.get_json()
        except:
            # Bad request as request body is not available
            # Add message for debugging purpose
            return "", 400

        record_created = messageCollection.insert(body)
        #
        # Prepare the response
        if isinstance(record_created, list):
            # Return list of Id of the newly created item
            return jsonify([str(v) for v in record_created]), 200
        else:
            # Return Id of the newly created item
            return jsonify({"id":str(record_created),
                            "content" : body["content"]
                            }), 200
    except:
        # Error while trying to create the resource
        # Add message for debugging purpose
        return "", 500


@app.route('/api/v1/messages/reply', methods=['POST'])
def reply_message():
    try:
        # Create new users
        try:
            body = request.get_json()
        except:
            # Bad request as request body is not available
            # Add message for debugging purpose
            return "", 400

    
        content = body["content"]
        content = ViTokenizer.tokenize(content)
        words = content.rsplit()
        words = [w for w in words if w not in STOP_WORDS]
        content = ''
        for w in words:
            content += w + ' '
        content = content[:-1]
        print(content)
        classified_content = keras_text_classifier.classify([content], label_dict=labels)
        print(classified_content)
        classified_content = classified_content[0]

        bot_reply = "Hi, I'm bot"
        if classified_content == "hello":
            bot_reply = random.choice(response["hello"])
        elif classified_content == "goodbye":
            bot_reply = random.choice(response["goodbye"])
        elif classified_content == "thanks":
            bot_reply = random.choice(response["thanks"])
        elif classified_content == "introduction":
            bot_reply = random.choice(response["introduction"])
        elif classified_content == "ask_weather":
            bot_reply = solve_question(content, is_general=True)["result"]
            if bot_reply == False:
                bot_reply = random.choice(response["cant"])
            else:
                bot_reply = random.choice(response["general_asking"]) +  bot_reply
        elif classified_content == "general_asking":
            bot_reply = solve_question(content, is_general=True)["result"]
            if bot_reply == False:
                bot_reply = random.choice(response["cant"])
            else:
                bot_reply = random.choice(response["general_asking"]) +  bot_reply
        elif classified_content == "math" or classified_content == "chemistry":
            bot_reply = solve_question(body["content"], is_general=False)["result"]
            if bot_reply == False:
                bot_reply = random.choice(response["cant"])
            else:
                bot_reply = random.choice(response["calculator"]) +  bot_reply
        else:
            bot_reply = solve_question(content, is_general=False)["result"]
            if bot_reply == False:
                bot_reply = random.choice(response["cant"])


        record_reply = messageCollection.insert({
            "content": bot_reply,
            "isBot": True,
            "time": time.time()
        })
        #
        # Prepare the response
        if isinstance(record_reply, list):
            # Return list of Id of the newly created item
            return jsonify([str(v) for v in record_reply]), 200
        else:
            # Return Id of the newly created item
            return jsonify(str(record_reply)), 200
    except:
        record_reply = messageCollection.insert({
            "content": random.choice(response["cant"]),
            "isBot": True,
            "time": time.time()
        })
        # Error while trying to create the resource
        # Add message for debugging purpose
        return "", 500


def solve_question(question, is_general=True):
    res = search_and_solve(question, is_general=is_general)
    return {
        "result": res["result"]
    }


@app.route('/api/v1/test', methods=['GET'])
def test():
    res = search_and_solve("Cân bằng phản ứng: H2+O2=H2O", is_general=False)
    return {
        "result": res["result"]
    }

if __name__ == '__main__':
    app.run(debug=True)
