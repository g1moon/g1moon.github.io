---
layout: post
title:  "미니 트위터에 Database 붙이기"
subtitle: "미니 트위터에 Database 붙이기"
categories: development
tags: web python
comments: true
---
	 
- [Python Flask로 미니 트위터 구현하기](https://zzsza.github.io/development/2019/03/02/mini-twitter-with-flask/)에서 구현한 코드를 디벨롭하는 글입니다
	- 메모리에 데이터를 올렸는데, 이젠 Database를 사용하려고 합니다
	
---

## 데이터베이스
- 데이터를 저장 및 보존하는 시스템으로 저장된 데이터를 읽거나 새로운 데이터를 저장, 기존의 데이터를 업데이트할 수 있음
- 관계형 데이터베이스(RDBMS)와 NoSQL로 불리는 비관계형 데이터베이스가 있음
- 관계형 데이터베이스
	- 데이터들이 서로 상호관련성을 가진 형태로 표현한 데이터
	- 대표적으로 MySQL, PostgreSQL
	- 모든 데이터들이 2차원 테이블로 표현되고, 컬럼과 로우로 구성됨
	- csv 파일
	- 로우는 고유 키(primay key)가 있고, 이 키를 통해 로우를 찾게 됨
	- 정규화
		- 중복을 최소화하도록 데이터를 구조화하는 프로세스 	
		- 여러 테이블에 정보를 나누어 저장하고 테이블끼리 관계를 설정해 연결하는 이유
			- 데이터의 중복 저장을 피할 수 있음 => 디스크 공간을 효율적으로 사용
			- 잘못된 데이터 저장을 피할 수 있음 => 중복을 최소화
	- 트랜잭션
		- 일련의 작업들이 하나의 작업처럼 취급되어 모두 다 성공하거나 모두 다 실패하는 것
		- 트랜잭션 기능을 위해 ACID 성질을 가짐
		- Atomicity, Concistency, Isolation, Durability (원자성, 일관성, 고립성, 지속성)
- 비관계형 데이터베이스
	- 데이터를 미리 정의할 필요가 없음
	- 스키마와 테이블 관계를 구현할 필요가 없고, 데이터가 들어오는 그대로 저장
	- MongoDB, Redis, Cassandra 등

### RDBMS vs NoSQL
- RDBMS의 장점
	- 데이터를 더 효율적이고 체계적으로 저장, 관리
	- 데이터들의 구조를 미리 정의해 데이터의 완전성이 보장
	- 트랜잭션 기능을 제공
- RDBMS의 단점
	- 테이블을 미리 정의해서 테이블 구조 변화 등에 덜 유연
	- 확장이 쉽지 않음
	- 서버를 늘려서 분산 저장하는 것이 쉽지 않음. 스케일 아웃(서버 수를 늘려서 확장)보단 스케일 업(서버의 성능을 높이는 것)으로 확장해야 함
- NoSQL의 장점
	- 데이터 구조를 미리 정의하지 않아도 되서 데이터 구조 변화에 유연
	- 시스템 확장하기가 비교적 쉽다. 스케일 아웃으로 확장 가능
	- 유연하고 확장하기 쉬워서 방대한 데이터 저장할 떄 유리
- NoSQL의 단점
	- 데이터의 완전성이 덜 보장
	- 트랜잭션이 안되거나 되더라도 비교적 불안정
	
---

### 맥에서 MySQL 설치하기
- [맥에서 MySQL 설치하기](https://zzsza.github.io/development/2018/01/18/Install-MySQL-mac/) 참고

### 스키마 구현하기
- mysql 접속

	```
	mysql -u root -p
	```

- DATABASE 생성

	```
	CREATE DATABASE miniter;
	```       

- DATABASE 사용

	```
	USE miniter;
	```
	
- Table 생성

	```
	CREATE TABLE users(
    id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    profile VARCHAR(2000) NOT NULL,
	    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	    updated_at TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
	    PRIMARY KEY (id),
	    UNIQUE KEY email (email)
	);
	
	CREATE TABLE users_follow_list(
	    user_id INT NOT NULL,
	    follow_user_id INT NOT NULL,
	    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	    PRIMARY KEY (user_id, follow_user_id),
	    CONSTRAINT users_follow_list_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id),
	    CONSTRAINT users_follow_list_follow_user_id_fkey FOREIGN KEY (follow_user_id) REFERENCES users(id)
	);
	
	CREATE TABLE tweets(
	    id INT NOT NULL AUTO_INCREMENT,
	    user_id INT NOT NULL,
	    tweet VARCHAR(300) NOT NULL,
	    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (id),
	    CONSTRAINT tweets_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id)
	);
	```
	
- Table 확인

	```
	show tables;
	explain users;
	explain users_follow_list;
	explain tweets;
	```
	
### SQLAlchemy를 사용해 연결하기
- 설치

	```
	pip3 install sqlalchemy
	pip3 install mysql-connector-python
	```	
	
- 설정 파일 생성(config.py)
	- 설정파일을 만드는 이유는 1) 설정 정보를 따로 관리해 민감한 정보를 노출하지 않기 위해 2) 각 환경에 맞는 설정 파일을 적용(local, Staging, Production)

	```
	db = {
    'user'     : 'root',
    'password' : '1234',
    'host'     : 'localhost',
    'port'     : 3306,
    'database' : 'miniter'
	}
	
	DB_URL = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
	```	

- Database 연결
	
	```
	def create_app(test_config=None):
		app = Flask(__name__)
		if test_config is None:
		    app.config.from_pyfile("config.py")
		else:
		    app.config.update(test_config)
		
		database = create_engine(app.config["DB_URL"], encoding="utf-8", max_overflow=0)
		app.database = database
		
		return app
	```

- CustomJSONEncoder 삭제
	- 이제 DB로 저장하기 때문에, SET 처리하는 Encoder가 필요없음 

### 회원 가입	
- 기존 코드
	
	```
	@app.route("/sign-up", methods=["POST"])
	def sign_up():
	    new_user = request.json
	    new_user["id"] = app.id_count
	    app.users[app.id_count] = new_user
	    app.id_count = app.id_count + 1
	
	    return jsonify(new_user)
	```	
	
- 리팩토링

	```
	@app.route("/sign-up", methods=["POST"])
	def sign_up():
	    new_user = request.json
	    new_user_id = app.database.execute(text("""
	        INSERT INTO users (
	          name,
	          email,
	          profile,
	          hashed_password
	        ) VALUES (
	          :name,
	          :email,
	          :profile,
	          :password
	        )
	    """), new_user).lastrowid
	
	    row = app.database.execute(text("""
	        SELECT
	            id,
	            name,
	            email,
	            profile
	        FROM users
	        WHERE id = :user_id
	    """), {
	        "user_id": new_user_id
	    }).fetchone()
	    
	    create_user = {
	        "id": row["id"],
	        "name": row["name"],
	        "email": row["email"],
	        "profile": row["profile"]
	    } if row else None
	    
	    return jsonify(create_user)
	```

- 실행할 파일 run.py 생성
	- 책에는 별도 설명이 나와있지 않음 

	```
	from app import create_app
	
	if __name__ == '__main__':
		app = create_app()
		app.run()
	```
	
- 실행

	```
	python3 run.py
	```
	
- http request

	```
	http -v POST localhost:5000/sign-up name="변성윤" email="zzsza@naver.com" password="1234" profile="Hi"
	```	

### Tweet
- 저장해야 하는 데이터를 HTTP Request를 통해 받아 데이터베이스에 저장
- 기존 코드

	```
    @app.route("/tweet", methods=["POST"])
    def tweet():
        payload = request.json
        user_id = int(payload["id"])
        tweet = payload["tweet"]

        if user_id not in app.users:
            return "사용자가 존재하지 않습니다", 400

        if len(tweet) > 300:
            return "300자를 초과했습니다", 400

        user_id = int(payload["id"])

        app.tweets.append({
            "user_id": user_id,
            "tweet": tweet
        })

        return "", 200
    ```
    
- 리팩토링 코드

    ```
    @app.route("/tweet", methods=["POST"])
    def tweet():
        user_tweet = request.json
        tweet = user_tweet["tweet"]

        if len(tweet) > 300:
            return "300자를 초과했습니다", 400

        app.database.execute(text("""
            INSERT INTO tweets (
              user_id,
              tweet
            ) VALUES (
              :id,
              :tweet
            )
        """), user_tweet)

        return "", 200
    ```
    
- HTTP Request

	```
	http -v POST localhost:5000/tweet id=1 tweet="First tweet~"
	```
	
### Timeline 
- SELECT 구문을 사용하는 엔드포인트
- 데이터를 읽어 JSON 형태로 변환해 HTTP Response로 보냄
- 기존 코드

	```
    @app.route("/timeline/<int:user_id>", methods=["GET"])
    def timeline(user_id):
	    if user_id not in app.users:
	        return "사용자가 존재하지 않습니다", 400
	
	    follow_list = app.users[user_id].get("follow", set())
	    follow_list.add(user_id)
	    timeline = [tweet for tweet in app.tweets if tweet["user_id"] in follow_list]
	
	    return jsonify({
	        "user_id": user_id,
	        "timeline": timeline
	    })
	
	return app
	```	
	
- 리팩토링 코드

	```
    @app.route("/timeline/<int:user_id>", methods=["GET"])
    def timeline(user_id):
        rows = app.database.execute(text("""
            SELECT
                t.user_id,
                t.tweet
            FROM tweets AS t 
            LEFT JOIN users_follow_list AS ufl
            ON ufl.user_id = :user_id
            WHERE t.user_id = :user_id
            OR t.user_id = ufl.follow_user_id
        """), {
            "user_id": user_id
        }).fetchall()

        timeline = [{
            "user_id": row["user_id"],
            "tweet": row["tweet"]
        } for row in rows]

        return jsonify({
            "user_id": user_id,
            "timeline": timeline
        })

    return app

	```

- HTTP Request

	```
	http -v GET localhost:5000/timeline/1
	```
	
### Follow
- 기존 코드

	```
    @app.route("/follow", methods=["POST"])
    def follow():
        payload = request.json
        
        user_id = int(payload["id"])
        user_id_to_follow = int(payload["follow"])

        if user_id not in app.users or user_id_to_follow not in app.users:
            return "사용자가 존재하지 않습니다", 400

        user = app.users[user_id]
        user.setdefault("follow", set()).add(user_id_to_follow)

        return jsonify(user)
	```

- 리팩토링 코드

	```
    @app.route("/follow", methods=["POST"])
    def follow():
        user_follow = request.json

        app.database.execute(text("""
            INSERT INTO users_follow_list (
              user_id,
              follow_user_id
            ) VALUES (
              :id,
              :follow
            )
        """), user_follow).rowcount

        return "", 200
	```	
	
- HTTP Request

	```
	http -v POST localhost:5000/follow id=1 follow=3
	```
		
### Unfollow
- 기존 코드

	```
    @app.route("/unfollow", methods=["POST"])
    def unfollow():
        payload = request.json

        user_id = int(payload["id"])
        user_id_to_follow = int(payload["unfollow"])

        if user_id not in app.users or user_id_to_follow not in app.users:
            return "사용자가 존재하지 않습니다", 400

        user = app.users[user_id]
        user.setdefault("follow", set()).discard(user_id_to_follow)

        return jsonify(user)
	```

- 리팩토링 코드

	```
    @app.route("/unfollow", methods=["POST"])
    def unfollow():
        user_unfollow = request.json

        app.database.execute(text("""
            DELETE FROM users_follow_list
            WHERE user_id = :id
            AND follow_user_id = :unfollow
        """), user_unfollow).rowcount

        return "", 200
	```	
	
- HTTP Request

	```
	http -v POST localhost:5000/unfollow id=1 unfollow=4
	```	
	
### 마무리
- 추후 리팩토링하면 좋을 것들
	- 함수화(insert, delete 하는 부분들)
	- ORM 사용 	
	
### Reference
- [깔끔한 파이썬 탄탄한 백엔드](http://www.yes24.com/Product/goods/68713424)
