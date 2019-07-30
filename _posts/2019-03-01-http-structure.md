---
layout: post
title:  "HTTP의 구조 및 핵심 요소"
subtitle: "HTTP의 구조 및 핵심 요소"
categories: development
tags: web
comments: true
---
	 
- HTTP의 구조 및 핵심 요소에 대해 정리한 글입니다
- 깔끔한 파이썬 탄탄한 백엔드 책을 참고했습니다

---

### HTTP
- HYPERTEXT TRANSFER PROTOCOL의 약자
- 웹상에서 서로 다른 서버간에 HTML을 서로 주고받을 수 있도록 만들어진 프로토콜, 통신 규약
	- 웹상에서 네트워크를 통해 통신할 때 어떤 형식으로 통신하자고 규정해놓은 "통신 형식" 혹은 "통신 구조"
- 프로토콜 중 가장 널리 사용되는 프로토콜이 HTTP

### HTTP 통신 방식
- 2가지 특징
- 1) HTTP의 요청(request)과 응답(response) 방식
	- 요청을 하고 응답을 받는 구조
	
	```
	@app.route("/ping", methods=["GET"])
	def ping():
		return "pong"
	``` 
- 2) stateless
	- 각 HTTP 통신은 독립적이며 그 전에 처리된 HTTP 통신에 대해 전혀 알지 못함
	- 상태를 저장할 필요가 없으므로 HTTP 통신 간의 진행이나 연결 상태의 처리, 저장을 구현 및 관리하지 않아도 됨
	- 그저 요청에 대해 독립적으로 응답만 보내줌
	- 단점 : stateless라 HTTP 요청시 필요한 모든 데이터를 매번 포함시켜 보내야 함  
		- 어떤 요청시 사용자가 로그인 되어야 한다고 가정할 때, 해당 사용자 로그인 여부를 포함해 HTTP 요청을 날려야 함
		- 이런 점을 해결하기 위해 쿠키(cookie)나 세션(session) 등을 사용해 필요한 진행 과정이나 데이터 저장
		- 쿠키 : 웹사이트에서 보내온 정보를 저장할 수 있도록 하는 작은 파일로 웹 브라우저(클라이언트)에 저장
		- 세션 : 쿠키와 마찬가지로 HTTP 통신상에 필요한 데이터를 저장하는데, 웹 서버에 데이터 저장 


## HTTP 요청 구조
- HTTP 요청 메세지는 아래와 같은 형태로 되어 있음

```
POST /payment-sync HTTP/1.1 #1

#2
Aceept: application/json
Accept-Encoding: gzip, deflate
Connection: keep-alive
Content-Length: 83
Content-Type: application/json
Host: intropython.com
User-Agent: HTTPie/0.9.3

#3
{
	"imp_uid": "imp_1234567890",
	"merchant_uid": "order_id_8237352",
	"status": "paid"
}	
```

- 1 : Start Line
- 2 : Header
- 3 : Body


### Start Line
- HTTP 요청의 시작줄
- search 엔드포인트에 GET HTTP 요청을 보내면 해당 HTTP 요청의 start line은 아래와 같음

	```
	GET /search HTTP/1.1
	```
	
- 구성 요소
	- HTTP 메소드
		- HTTP 요청이 의도하는 액션을 정의하는 부분으로 GET, POST, PUT, DELETE 등 
	- Request target
		- 해당 HTTP 요청의 목표 주소, 위 예제에선 search 엔드포인트에 보내는 HTTP 요청의 request target은 /search 
	- HTTP version 	
		- 해당 요청의 HTTP 버전
		- 현재 1.0, 1.1, 2.0이 있음
		- 버전에 따라 요청 메세지 구조나 데이터가 다를 수 있기 때문에 버전을 명시

### Header 
- HTTP 요청 그 자쳉 대한 정보를 담고 있음
	- 예를 들어 Request 메세지의 전체 크기(Content-Length) 같은 정보
	- 파이썬의 dictionary처럼 key와 value로 되어 있고, :로 연결됨
- 자주 사용되는 헤더
	- Host
		- 요청이 전송되는 target의 호스트 URL 주소
		- Host: google.com
	- User-Agent
		- 요청을 보내는 클라이언트의 정보, 예를 들면 웹 브라우저
		- User-Agent: Mozila/5.0
	- Accept
		- 해당 요청이 받을 수 있는 response body 데이터 타입
		- MIME(Multipurpose Internet Mail Extensions)  type이 value로 지정
			- 예 : application/json, text/csv, text/html, image/jpen, application/xml
		- Connection
			- 해당 요청이 끝난 후에 클라이언트와 서버가 계속 네트워크 연결을 유지할지, 끊을지
			- 매 요청마다 HTTP 연결을 새로 만드는 것은 시간이 걸리기 때문에 처음 만든 연결을 재사용하는 것을 선호하기도 함
			- keep-alive면 계속해서 요청을 보낼 예정이니 네트워크 연결을 유지
			- close면 네트워크 연결을 닫아도 된다
		- Content-Type
			- 메세지 Body의 타입을 알려주며 Accept 헤더와 마찬가지로 MIME type 사용
			- 예 : Content-Type: application/json
		- Content-Length
			- 요청이 보내는 메세지 body의 총 사이즈
			- 예 : Content-Length: 257

### Body
- HTTP 요청이 전송하는 데이터를 담고 있는 부분
- 전송하는 데이터가 없으면 Body는 비어있음      


---


## HTTP 응답 구조
- 요청 메세지처럼 응답도 크게 3부분으로 구성됨

```
#1
HTTP/1.1 404 Not Found

#2
Connection: close
Content-Length: 1573
Content-Type: text/html; charset=UTF-8
Date

#3
<DOCTYPE html>
<html lang=en>
~~~~~

```

- 1 : Status Line
- 2 : Headers
- 3 : Body


### Status Line
- HTTP 응답 메세지 상태를 간략히 요약해주는 부분
- HTTP version, Status Code, Status Text로 나타남

### Header
- HTTP 요청의 헤더와 동일
- 단, HTTP Response시에만 사용되는 헤더가 있음
	- HTTP Response엔 User-Agent 헤더 대신 Server 헤더가 사용

### Body
- HTTP 요청 메세지의 body와 동일
- 전송하는 데이터가 없으면 body 부분은 비어있게 됨


---

## 자주 사용되는 HTTP 메소드
### GET
- 어떤 데이터를 서버로부터 요청(GET)할 때 주로 사용
- 데이터의 생성, 수정, 삭제 등의 변경사항 없이 단순히 데이터를 받아오는 경우 사용
- 데이터를 받아올 때 사용되서 HTTP Response의 body가 비어있는 경우가 많음


### POST
- 데이터를 생성하거나 수정, 삭제 요청을 할 때 주로 사용되는 메소드

### OPTIONS
- 특정 엔드포인트에서 허용하는 메소드들이 무엇이 있는지 알고자 할 경우 사용
- Flask에선 OPTIONS 메소드를 자동으로 구현해줌


---

## API 엔드포인트 아키텍처 패턴
- 크게 2가지로 REST 방식과 GraphQL

### RESTful HTTP API
- API에서 전송하는 리소스(resource)를 URI로 표현하고 해당 리소스에 행하고자 하는 의도를 HTTP 메소드로 정의
- /users라는 엔드포인트에서 사용자 정보를 받아오는 요청

```
HTTP GET /users
GET /users
```

- 새로운 사용자를 생성하는 엔드포인트는 URI로 /user로 정할 경우

```
POST /user
{
	"name" : "변성윤".
	"email" : "zzsza@naver.com"
}
```

- 장점
	- 자기 설명력 : 엔드포인트의 구조만 봐도 엔드포인트가 제공하는 리소스와 기능을 파악할 수 있음
- 단점
	- API 구조가 특정 클라이언트에 맞춰져서 다른 클라이언트에서 사용하기엔 적합하지 않음
	- 예 : 페이스북 웹용 API가 모바일에선 사용하기 적합하지 않아 새로 만듬
 
### GraphQL
- 페이스북에서 만듬
- 엔드포인트가 오직 하나고, 엔드ㅗ인트에 클라이언트가 필요한 것을 정의해서 요청하는 식
- REST 방식과 반대!
	- 서버가 정의한 틀에서 클라이언트가 요청하는것이 REST식, 클라이언트가 필요한 것을 서버에 요청하는 방식이 GraphQL
- ID가 1인 사용자 정보와 친구들의 이름 정보를 API로 받아올 경우, REST 방식에선 아래와 같이 2번 요청을 보내야 함

	```
	GET /users/1
	GET /users/1/friends
	```

- 한번으로 줄이려면 아래처럼

	```
	GET /users/1?include=friends.name
	```	

- GraphQL을 사용하면 다음과 같이 HTTP 요청

	```
	POST /graphql
	
	{
	 user(id: 1) {
	     name
	     age
	     friends {
	       name
	     }
	   }
	}
	```


- GraphQL은 장점이 많지만 REST에 비해 오래되지 않은 기술이라 널리 사용되고 있진 않음. 백엔드 개발 입문자는 REST API를 먼저 이해하는 것이 더 좋음
