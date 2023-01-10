from flask import Flask, request, jsonify

app = Flask(__name__)

# server resource
resource = []


# 사용자 정보 조회
@app.route('/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    for user in resource:
        if user['user_id'] is user_id:
            return jsonify(user)
    return jsonify(None)


# 사용자 추가
@app.route('/user', methods=['POST'])
def add_user():
    user = request.get_json()  # HTTP 요청의 body에서 json 데이터를 불러옴
    resource.append(user)  # 리소스 리스트에 추가
    return jsonify(resource)


if __name__ == "__main__":
    app.run(debug=True)
