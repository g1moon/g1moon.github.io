---
layout: post
title:  "시스템 모델링 개념"
subtitle: "시스템 모델링 개념"
categories: data
tags: simulation
comments: true
---


- 시스템 모델링 개념에 대한 글입니다


---

### 시스템 정의 및 분류
- 시스템 및 복합 시스템(System of Systems)
	- 시스템 정의(출처 : IEEE)
		- 여러 구성 요소(Component)들이 모여서 개별 구성요소들만으로는 불가능했던 기능을 만족시키는 구성 요소들의 조합
	- 복합 시스템의 정의(System of Systems: SoS)
		- 시스템 정의에서 시스템을 구성하는 구성 요소들이 독립적인 역할을 수행하는 시스템인 경우 이들의 조합
- 시스템 계층구조 및 구성요소들
	- 시스템 및 구성요소 : 미 국방표준 MIL-STD-499B
	- <img src="https://www.dropbox.com/s/jbgoe742a4qwtny/Screenshot%202019-06-02%2022.11.44.png?raw=1">
- 시스템 구성 예시
	- 입출력 및 구성 요소는 M&S 목적에 따라 달라짐!
	- <img src="https://www.dropbox.com/s/z9clzfnmtkovagp/Screenshot%202019-06-02%2022.13.12.png?raw=1">
- 추상화 수준에 따른 시스템 분류
	- 시스템 동작을 미시적으로 묘사할 경우 시스템 자체 지능이 거의 없음
	- 시스템 동작을 거시적으로 묘사할 경우 시스템 자체 지능이 있음 
	- <img src="https://www.dropbox.com/s/7saoc9740qazqvq/Screenshot%202019-06-02%2022.18.44.png?raw=1">
	- 연속 시간 시스템
		- 시간의 모든 값에서 관측하고 상태를 모두 봄
		- 세부적 관찰
		- 유년기에 세세하게 관찰
		- 미분 방정식
		- 아날로그 시스템
	- 이산 사건 시스템
		- 가로축의 시간을 무시하고 사건이 일어날 때만 시스템의 상태를 관측
		- 유년기 이후 성인이 되면 특정 이벤트(아프거나 입원헀다) 등으로 관측해도 충분
		- DEVS 형식론
		- 생산 시스템
	- 이산시간 시스템
		- 특별한 시간에 대해서 관측
		- 아침, 점심, 저녁
		- 하루 단위 등
		- 차분 방정식
		- 샘플데이터 시스템
	- 디지털 시스템
		- 관측하는 값을 이상화
		- 상태 변수의 값(ex: 체온) 36.7도처럼 수치화
		- 유한 상태 기계
		- VLSI 시스템
- 시스템을 추상화 수준으로 계층화시킨 예
	- 로봇 시스템
		- Planner/Scheduler
			- 무엇을 할건지? 명령들 
		- Controller
			- 어떻게 할건지? 전기 신호들 
		- Manipulator  
		- <img src="https://www.dropbox.com/s/gzxzi6cyfso3zme/Screenshot%202019-06-02%2022.28.18.png?raw=1">


---

### 동적 시스템 모델 분류
- 시스템 : 정적 시스템 vs 동적 시스템
	- 상태 변수와 시간의 관련성에 따라 나뉨
	- <img src="https://www.dropbox.com/s/fih6baar2tbug17/Screenshot%202019-06-02%2022.53.35.png?raw=1">
- 동적 시스템 모델 분류
	- 상태변수 갱신 시점에 따른 분류
	- 상태변수 갱신의 불확실성에 따른 분류
	- 상태변수들 사이의 관계식 표현 방법에 따른 분류
	- <img src="https://www.dropbox.com/s/4858x8gn89sr8pf/Screenshot%202019-06-02%2022.56.57.png?raw=1">
- 연속, 이산사건 및 하이브리드 모델
	- <img src="https://www.dropbox.com/s/x9quw20h127dxze/Screenshot%202019-06-02%2023.00.55.png?raw=1">
	- 이산사건 모델 : 메세지가 들어가면 시스템이 움직이고, 메세지가 없으면 메세지가 오길 기다림
- 확정적 모델 vs 확률적 모델
	- 상태를 바꿀 때 확정적인 값으로 하느냐 확률로 표현하느냐
	- 확정적 모델은 확률이 1인 경우만 고려한 케이스
	- <img src="https://www.dropbox.com/s/nzx5thphlnzz893/Screenshot%202019-06-02%2023.06.47.png?raw=1">
- 해석적 모델 vs 시뮬레이션 모델
	- 해석적 모델 : X, Y, Q 사이의 관계식이 닫힌 형태의 수식으로 표현(미분 방정식)
	- 시뮬레이션 모델 : X, Y, Q 사이의 관계식을 논리 형태로 표현
	- 시뮬레이션 모델은 특정한 가정을 적용한 해석적 모델로 변환할 수 있음
	- <img src="https://www.dropbox.com/s/0hhjebc71jjox5t/Screenshot%202019-06-02%2023.12.26.png?raw=1">
- 형식론적 모델 vs 비형식론적 모델
	- 형식적(수학적) 모델 : X, Y, Q 사이의 관계식을 수학적 논리로 표현
		- DEVS 모델
	- 비 형식론적 모델 : X, Y, Q 사이의 관계식을 언어적 논리로 표현
		- Event-oriented 모델, 사건 기반
	- 형식론적 모델을 비 형식론적 모델로 변환 가능
	- <img src="https://www.dropbox.com/s/2yxgf8dl2ohvy72/Screenshot%202019-06-02%2023.14.27.png?raw=1">

---

### 시스템 모델 명세 틀
- 모델링 틀, 형식론 및 방법론
	- 모델링 틀(Framework)
		- 모델을 완전하고 애매모호함 없이 명세할 수 있는 공식/템플릿
		- 신문기사 작성(모델링) 틀 : 육하원칙(5W1H)
	- 모델링 형식론(Formalism) = 수학적 형식론
		- 수학적으로 표현된 모델링 틀
		- 연속시간 모델링 형식론 : (편)미분 방정식
	- 모델링 방법론(Methodology)
		- 검증된 원리와 가정을 이용한 일반화된 모델링 절차와 방법
		- 생산시스템 모델링: DEVS 모델링 방법론
- 모델 개발 시 수학적 모델링 틀의 역할
	- 경우의 수 시뮬레이션 문제
	- 비 형식론적 접근은 모델이 없고 시뮬레이션만 있음
	- 형식론적 접근은 모델링과 시뮬레이션이 명시적으로 분리되어 있음
	- 수학적 모델의 역할
		- 완전성(Completeness)
		- 검증성(Testability)
		- 통신 수단(communication means)
		- 수학적 처리(Mathematical manipulation)
		- 일반성(Generality)
	- <img src="https://www.dropbox.com/s/tg41ap88ecd4u7w/Screenshot%202019-06-02%2023.41.50.png?raw=1">
- 신문기사 작성(모델링) 틀 : 5W1H
	- <img src="https://www.dropbox.com/s/u3s8juxipt5de5f/Screenshot%202019-06-02%2023.47.20.png?raw=1">
- 이산사건 시스템 모델링 틀 : 3S4F (DEVS)
	- 만들기 매우 어려움
	- 3개의 집합과 4개의 함수
	- <img src="https://www.dropbox.com/s/4xur5qm9rntmutg/Screenshot%202019-06-02%2023.51.57.png?raw=1">
- 모델링 틀 및 방법론 적용의 의의
	- 모델 개발 전문가와 일반 모델 개발자가 만든 것이 대등한 산출물을 낼 수 있음
	- 모델링 틀과 개발 절차 방법을 적용
	- 방법론 지원, 도구, 환경을 적용
	- <img src="https://www.dropbox.com/s/iyx148qt78l7z7n/Screenshot%202019-06-02%2023.53.17.png?raw=1">    

---

### 의미론적 표현 : 수학, 전산학, M&S 공학
- 의미론 표현 : 수학 vs 전산학 vs M&S 공학
	- <img src="https://www.dropbox.com/s/ys7bv8yv90nbrjl/Screenshot%202019-06-02%2023.57.19.png?raw=1">
- 의미론 해석 - 수학
	- <img src="https://www.dropbox.com/s/m8vxw7spuw9hk1o/Screenshot%202019-06-02%2023.58.25.png?raw=1">  
- 의미론 해석 - 전산학
	- <img src="https://www.dropbox.com/s/cuqbn1atj6xtqk0/Screenshot%202019-06-02%2023.59.20.png?raw=1">
- 의미론 해석 - M&S 공학
	- 상태가 입력이 없어도 바뀌는 경우(나이) 입력이 없을시 상태천이가 가능한 델타 함수 정의
	- <img src="https://www.dropbox.com/s/r4mipubbdhsfyc0/Screenshot%202019-06-03%2000.00.05.png?raw=1">
- 의미론적 표현법의 사용 예
	- <img src="https://www.dropbox.com/s/rta2zy53jrcd6aj/Screenshot%202019-06-03%2000.01.18.png?raw=1">    



### Reference
- KAIST 김탁곤 교수님의 [시스템 모델링 시뮬레이션 입문](https://kooc.kaist.ac.kr/isms1)


 
