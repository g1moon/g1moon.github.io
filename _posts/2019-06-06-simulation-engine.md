---
layout: post
title:  "시뮬레이션 엔진의 개념 및 DEVS 알고리즘 적용"
subtitle: "시뮬레이션 엔진의 개념 및 DEVS 알고리즘 적용"
categories: data
tags: simulation
comments: true
---


- 시뮬레이션 엔진의 개념 및 DEVS 알고리즘 적용에 대한 글입니다


---

### 시뮬레이션 엔진 개념 및 시간 진행 관리
- 엔진 : 자동차 vs 시뮬레이션
	- 유사성이 존재
	- <img src="https://www.dropbox.com/s/u04ymch20w2iusq/Screenshot%202019-06-05%2023.42.08.png?raw=1">
- 엔진과 연료의 분리 및 다양성 : 자동차 vs 시뮬레이터
	- <img src="https://www.dropbox.com/s/feuxw088vkbnl7a/Screenshot%202019-06-05%2023.42.40.png?raw=1">
- 시뮬레이터(모델+엔진)
	- 모델과 시뮬레이션 엔진의 분리
	- 시뮬레이션 엔진에선 시간을 관리
	- <img src="https://www.dropbox.com/s/p2yw026lh7n2o7w/Screenshot%202019-06-05%2023.46.35.png?raw=1"> 
- 시뮬레이션 알고리즘(모델 실행 엔진)의 기본 틀
	- t : 현재 시각(또는 시뮬레이션 시각)
		- 프로그램 변수인지 실제 시각인지?
		- 가상 시간 시뮬레이션 / 실시간 시뮬레이션
	- 종료 조건은 어떻게 정하는지?
		- 시간 혹은 변수 값
		- 변수값이 특정 임계값 이상이면 종료 등  
	- 어떤 모델을 어떻게 실행할 것인가
		- 동시에 여러개, 한번에 한번만
	- T1은 상수인지 난수인지?
		- 누가 어떻게 정하는지?  
	- <img src="https://www.dropbox.com/s/6mdad6matxzc5c8/Screenshot%202019-06-05%2023.49.00.png?raw=1">
- 시간 진행 관리 개념 : 1개의 모델
	- <img src="https://www.dropbox.com/s/49706j1s3040frc/Screenshot%202019-06-05%2023.55.07.png?raw=1">
- 시간 진행 관리 개념 : 주기가 동일한 2개의 모델
	- <img src="https://www.dropbox.com/s/fi9q2xkjq5q6kad/Screenshot%202019-06-05%2023.56.18.png?raw=1">
- 시간 진행 관리 개념 : 주기가 다른 2개의 모델
	- <img src="https://www.dropbox.com/s/khlkmgqepky1a93/Screenshot%202019-06-05%2023.59.06.png?raw=1">

---

### 모델 종류에 따른 시뮬레이션 엔진
- 연속시간 모델, 이산사건 모델 및 하이브리드 모델의 시뮬레이션 엔진 소개
- 시뮬레이션 알고리즘(엔진)의 기본 틀
	- <img src="https://www.dropbox.com/s/hi2gwimias1oh7i/Screenshot%202019-06-06%2000.01.52.png?raw=1">
- 시뮬레이션 엔진의 3가지 기능
	- 사용자 명세
	- M&S 개발환경 제공
	- 시뮬레이션 엔진의 기능
		- 모델 해석
		- 시간 진행 관리
		- 모델 선택/데이터 전달
	- <img src="https://www.dropbox.com/s/dw4f2nvpozws5aq/Screenshot%202019-06-06%2000.02.36.png?raw=1">
- 연속시간 모델의 시뮬레이션 엔진 기능
	- 출력에서 입력의 인과 관계가 매 루프마다 진행
	- <img src="https://www.dropbox.com/s/tn04j2haqqc3zpb/Screenshot%202019-06-06%2000.58.09.png?raw=1">     
- 이산사건 모델의 시뮬레이션 엔진 기능
	- 모델 해석 : 모델에 정의된 함수 호출
	- 시간 진행 관리 : 사건 발생 시간순으로 정리(사건 발생 시간은 모델에서 알려줌)
	- 모델 선택 / 데이터 전달 : 매 루프마다 다음 사건발생 시각이 최소인 모델을 선택하고 이 모델과 연결관계에 있는 모델에 데이터 전달
	- <img src="https://www.dropbox.com/s/amn3b0grtevurbv/Screenshot%202019-06-06%2001.09.09.png?raw=1">
- 하이브리드 시뮬레이션 엔진의 한가지 형태
	- 연속시간 시뮬레이터 + 이산사건 시뮬레이터
	- 하이브리드 시뮬레이터 엔진
		- 시간 진행 관리 : 두 시뮬레이터 사이의 시간 진행 동기화(연속시간 모델의 선행 시뮬레이션 필요)
		- 데이터 변환 : 서로 다른 형태의 데이터간 변환 필요, SE 변환기 : 연속 신호(S) => 이산사건(E) 
	- <img src="https://www.dropbox.com/s/8cf43knoeli6lbt/Screenshot%202019-06-06%2001.11.08.png?raw=1">
- 시뮬레이션 시간 진행 : 연속 시간 vs 이산사건 모델
	- 시간 진행의 차이
	- 연속시간은 일정한 시간 간격으로 모델에 내려주면 그에 따라 모델이 시뮬레이션 진행함
		- 1개의 메세지로 시간 진행 
	- 이산사건은 모델이 Ti를 알려주고 t를 t+Ti로 갱신 후 모델 실행 명령
		- 2개의 메세지로 시간 진행 
	- <img src="https://www.dropbox.com/s/5dan9hkfop5c291/Screenshot%202019-06-06%2001.16.12.png?raw=1">
- 모델과 엔진의 시간진행 방법이 다른 경우 문제점
	- 이산시간 시뮬레이션 엔진
	- T1 시점이 우리가 의도할 때가 아닐 수 있음
	- <img src="https://www.dropbox.com/s/zrx3keh633phmc0/Screenshot%202019-06-06%2001.18.18.png?raw=1"> 

---
	
### DEVS 모델의 시뮬레이션 엔진 구조 및 원자모델 시뮬레이션 알고리즘
- DEVS 모델의 계층적 시뮬레이션의 특징
	- 계층적 시뮬레이터 구조 = DEVS 모델 구조
		- 원자모델 AM <-> 원자 모델 실행기 S:AM
		- 결합모델 CM <-> 결합 모델 실행기 C:CM
	- 계층적 시뮬레이션 알고리즘
		- 시뮬레이션 엔진의 3대 기능 수행
		- 모델 해석, 시간 진행 관리, 모델 선택 및 데이터 전달
	- 분산 알고리즘
		- 전역 정보를 가지고 있지 않음
		- 순차적 컴퓨팅 환경 및 분산 / 병렬 환경에서 구현 가능
- DEVS 모델의 계층적 시뮬레이션 엔진 개념
	- 실행기 : DEVS 모델의 함수 호출
	- <img src="https://www.dropbox.com/s/gi023g8kkwbqt3q/Screenshot%202019-06-06%2009.52.38.png?raw=1">
- 모델 실행기의 DEVS 함수 호출 시점
	- <img src="https://www.dropbox.com/s/8g0lpq6cf8z1ms1/Screenshot%202019-06-06%2009.54.29.png?raw=1"> 
- 계층적 시뮬레이션 엔진 구조
	- 모델 + 실행기로 구성
	- <img src="https://www.dropbox.com/s/dm1gsl9xdfj71kn/Screenshot%202019-06-06%2009.57.28.png?raw=1">
- 모델 실행기의 입/출력 정의
	- <img src="https://www.dropbox.com/s/0hjfcexexsci9lj/Screenshot%202019-06-06%2009.58.52.png?raw=1">
- 원자 모델 실행기 알고리즘 개요
	- wait하다 외부 상태 천이, ta로 다음 시간 계산 -> 다시 전달 -> tn이 다되면 star t -> 내부 상태천이 및 tn 계산
	- <img src="https://www.dropbox.com/s/o909x819zhr54bo/Screenshot%202019-06-06%2010.00.54.png?raw=1">
- 원자 모델 실행기 알고리즘의 수도 코드
	- when_receive 함수
	- <img src="https://www.dropbox.com/s/6d2tbrepvspp6e9/Screenshot%202019-06-06%2010.01.59.png?raw=1">


---

### DEVS 결합 모델 시뮬레이션 알고리즘
- DEVS 결합 모델의 시뮬레이션 알고리즘을 이해하기
- 결합모델의 다음 시뮬레이션 시각 tn 결정 방법
	- 어떻게 시각이 결정되는지 알아야할 필요가 있음
	- tn= min{t+ta(q1), t+ta(q2)}
	- <img src="https://www.dropbox.com/s/vczzba17w3bhyh0/Screenshot%202019-06-06%2010.35.47.png?raw=1">
- 결합 모델 실행기 알고리즘 개요
	- 결합 모델과 실행기는 통신을 하며 실행기가 호출
	- 원자모델 실행기를 컴포넌트로 가지고 있는데 실제 xt는 원자 모델이 x를 받아 출력과 입력 상태를 전이하기 때문에  이 xt를 누가 받을지 결합 모델이 선별해야 함 => M*
	- 동시에 시뮬레이션을 할 경우 SELECT 함수를 사용해 우선순위를 가지고 있는 모델을 선택함 => 출력 발생 후 내부 상태 천이
	- <img src="https://www.dropbox.com/s/xoe67fr3m2d10b4/Screenshot%202019-06-06%2011.17.54.png?raw=1">
- 결합 모델 실행기 알고리즘 수도 코드
	- influences : M2는 M1로부터 메세지를 받는 모델 
	- <img src="https://www.dropbox.com/s/lkxrmwlbvq4gn7w/Screenshot%202019-06-06%2011.18.53.png?raw=1">
- 스케줄 충돌 - 동일한 시각에 2개 이상의 이벤트 발생하는 경우
	- in 이벤트는가 M이 제어할 수 없고 M의 출력으로 보낸 모델의 타임아웃에 의해 제어됨
	- out 이벤트는 M의 time out에 의해 예약됨
	- 모델이 여러가지 존재하면 이런 상황이 많을 수 있음
	- 1) 입/출력 이벤트 동시 발생
	- 2) 출력 이벤트 동시 발생
		- DEVS에선 일어나지 않음. 출력은 함수(1:1)로 정의됨 
	- 3) 입력 이벤트 동시 발생
	- DEVS 모델, 이산사건 모델에서 실제의 이벤트가 미리 스케쥴된다는 것은 출력을 스케쥴한다고 이해하면 됨! (입력은 스케쥴할 수 없음)
	- <img src="https://www.dropbox.com/s/ed0kyzzwb0dgzd7/Screenshot%202019-06-06%2011.22.49.png?raw=1">
- 스케쥴 충돌시 우선순위 명세의 중요성
	- 우선순위 명세는 실제 순위와 일치하도록 명세
	- <img src="https://www.dropbox.com/s/651ql40i3ofstjy/Screenshot%202019-06-06%2011.32.14.png?raw=1"> 	
- DEVS 형식론에서 Select 함수의 불편한 진실
	- 복잡한 우선순위 명세를 거쳐야 함
	- DEVS 형식론의 집합론 기반과 계층적 모델 명세의 한계(순서를 정할 수 없음)
	- DEVS M&S 환경에선 API를 사용해 모든 모델의 우선순위를 한번에 명세
	- <img src="https://www.dropbox.com/s/1ezz12z1bxo0yx8/Screenshot%202019-06-06%2011.36.42.png?raw=1">

 

### Reference
- KAIST 김탁곤 교수님의 [시스템 모델링 시뮬레이션 입문](https://kooc.kaist.ac.kr/isms1)


 
