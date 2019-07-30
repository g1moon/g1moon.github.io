---
layout: post
title:  "모델링 시뮬레이션(M&S)의 기초"
subtitle: "모델링 시뮬레이션(M&S)의 기초"
categories: data
tags: simulation
comments: true
---


- 모델링 시뮬레이션 기초에 대한 글입니다


### M&S 기본 개념
- Modeling + Simulation
	- Modeling
		- 대상 시스템의 동작 특성 명세서로 방정식, 그래프, 테이블, 규칙 등으로 표현
	- Simulation
		- 명세된 모델을 실행시키는 가상 실험
- M&S와 컴퓨터 프로그램 관계
	- M&S는 컴퓨터 프로그래밍도 아니고 SW 공학도 아님
	- 컴퓨터 시뮬레이션에서는 명세된 모델을 컴퓨터 프로그램으로 구현한 후 시뮬레이션 알고리즘을 사용해 모델을 실행함
	- 시뮬레이션이 반드시 컴퓨터를 사용해야 하는 것은 아님
- 모델링과 시뮬레이션 방법과의 관계
	- 모델링 : 방정식을 세우는 과정
	- 시뮬레이션 알고리즘 : 방정식을 푸는 과정
- M&S 정확도와 결과의 신뢰도
	- M&S의 정확도 vs 결과의 신뢰도
	- 모델(데이터+연산 논리)의 정확도가 높고, 시뮬레이션(모델 실행 알고리즘)의 정확도가 높아야 M$S 결과에 대한 신뢰도도 높다고 할 수 있음
- M&S는 문제 해결 도구
	- M&S 프로세스 : 문제 => 모델링 => 시뮬레이션
		- 시뮬레이션의 결과가 문제의 답임 
		- <img src="https://www.dropbox.com/s/rps0h0udipz9uf2/Screenshot%202019-06-02%2020.13.58.png?raw=1">
- M&S 적용 효과는?
	- 과학적 의사 결정 : 의사 결정 결과에 대한 신뢰성 증대
	- 효율적 업무 수행 : 비용과 시간 절감
	- 안전한 시스템 운용 : 가상적 모의 실험으로 위험성 감소
- M&S 결과의 신뢰성 전제 조건
	- 모델 검증(V&V : Verification and Validation)
		- 우리가 만든 모델이 실제랑 동일해야 함
		- 검증은 모델링 목적에 부합하는 데이터에 국한해야 함
		- 모델링 목적을 만족시키는 검증 
			- <img src="https://www.dropbox.com/s/sogn0ji2vujl9kr/Screenshot%202019-06-02%2020.19.59.png?raw=1">
- V&V가 안된 M&S 적용시 문제점
	- 검증되지 않는 모델을 사용한 시뮬레이션의 결과
	- 1) M&S 결과 신뢰도 X
	- 2) 의사 결정의 혼란
	- 따라서 실제 체계와 비교가 필수
	- 모델 검증을 위해서 운영중인 시스템의 데이터 수집이 필수
- M&S 관련 용어 이해 : 유추적 의미로 접근
	- <img src="https://www.dropbox.com/s/h6y6pw02l3vm2ev/Screenshot%202019-06-02%2020.27.55.png?raw=1">
- M&S 공학 수명 주기 및 학술 분야 적용 기술
	- <img src="https://www.dropbox.com/s/ru9s8adot5sfbln/Screenshot%202019-06-02%2020.28.18.png?raw=1">


---


### 목적 지향적 모델링 및 M&S의 3 요소
- 목적 지향적 모델링
	- 목적 지향적 지도그리기
	- 목적에 따라 모델이 다를 수 있음
- 모델링 시 고려 사항
	- 모델의 추상화 레벨
		- 무엇을 반영하고 무엇을 무시할 모델인지의 문제 
	- 모델의 해상도
		- 모델의 구조 및 상태 변수 값을 나타내는 상세도 
	- 모델의 다측면성
		- 시스템에 대한 관찰, 생각하는 관점 
	- 목적 지향적(Objective-driven) M&S
		- 동일한 시스템에 대해서도 모델링 목적에 따라 모델이 달라짐
- 목적 지향적 모델링 예 (1)
	- 구조가 달라지는 경우
	- 컴퓨터 시스템을 만들어보자
		- <img src="https://www.dropbox.com/s/yhbqgey1ghugvwi/Screenshot%202019-06-02%2020.37.22.png?raw=1">
- 목적 지향적 모델링 예 (2)
	- 변수가 달라지는 경우 + 목적에 따라 달라지는 경우
	- <img src="https://www.dropbox.com/s/boots3v5961edai/Screenshot%202019-06-02%2020.40.30.png?raw=1">
- M&S의 목적 : 기능 vs 성능/효과도
	- 모델의 목적을 여러가지를 동시에 수용하면 모델의 복잡도가 높아지고 ->  정확도가 떨어지고 -> 시뮬레이션 결과의 신뢰도가 떨어짐
	- <img src="https://www.dropbox.com/s/vltkfh5y6fxvneb/Screenshot%202019-06-02%2020.43.50.png?raw=1">
- M&S의 3 요소
	- 대상 시스템
	- 모델 : 시스템에서 모델 설계
	- 시뮬레이터 : 모델 구현
- M&S 3 요소 사이들 관계 : V&V
	- 동적 실증(Dynamic Validation) : 실제 시스템이 시뮬레이터와 일치하는가?
	- 정적인 실증(Static Validation) : 구현되기 전에 모델 설계 문서를 통해 실시
		- 실행 불가로 인해 실증 한계점 존재 
	- 검증(Verification) : 모델과 시뮬레이터가 일치하는지? 모델이 구현된 후 실시 
		- 코드 디버깅과 동일한 프로세스
	- <img src="https://www.dropbox.com/s/08oqpw3rhcj3r0q/Screenshot%202019-06-02%2020.50.59.png?raw=1">


---


### 모델 명세 및 실행 방법
- 모델링
	- 사람이 하는 일
	- 모델 : 실제 시스템의 동작을 법칙화해서 표현한 동작 명세서(Specification)
	- 모델링 틀 : 동작 명세서의 Template => 방정식, 규칙, 알고리즘 등
	- 모델의 다측면성 : 대상 시스템의 도메인 지식을 변환해 모델링 틀에 맞게 채우는 과정 (매우 어려운 과정..!)
		- 생산 시스템의 모든 과정을 이해하고 구현해야 함
- 모델링 과정
	- 실제 시스템의 동작을 나타내는 다양한 소스 : 교과서, 매뉴얼, 각종 법규/규정, 운영 경험/지식
	- 방정식, 알고리즘, 규칙
		- Input : 가상 시나리오
		- Output 대응하는 결과
	- 모델링 = 지식 변환 과정
		- 도메인 전문가와 M&S 전문가가 지식 교환 및 변환
	- 모델 Validation 필수!
	- <img src="https://www.dropbox.com/s/7flbhpesshicp31/Screenshot%202019-06-02%2020.55.30.png?raw=1">
- 모델 명세 방법
	- 1) 수식적 틀에 의한 명세
		- 수학 방정식
			- 1차 방정식
			- 3차 편미분 방정식(비행기 기독 방정식)  
	- 2) 비수식적 틀에 의한 명세 
		- 규칙
			- 규칙 1번 : 만약 비가 오면 집과 회사까지 1시간 소요
			- 회사 인사 규칙(승진, 보직, 정년 등 규정 모음집)
- 모델 명세 방법 예
	- 수식적 모델
		- Input : x
		- Output : y 
		- y=2x+3   
	- 비 수식적 모델
		- Input : 일기 예보
		- Output : 출퇴근 소요 시간 
		- 만약 비가 오면 30분 걸림
		- 만약 눈이 오면 50분 걸림 
- 시뮬레이션 엔진 : 모델 해석(실행)하기
	- 시뮬레이션 엔진 = 모델 실행기 = 모델 해석기 = 모델 실행 알고리즘
	- 모델(예: 방정식)을 실행시키는(예: 방정식 풀기) 알고리즘
	- <img src="https://www.dropbox.com/s/d2gemklyaa8i2rx/Screenshot%202019-06-02%2021.01.16.png?raw=1">
- 시뮬레이터
	- 모델 + 시뮬레이션 엔진(모델 실행기) + 시뮬레이션에 필요한 기타 자원의 집합체
	- 아주 간단한 경우 소프트웨어만 시뮬레이터라 할 수 있고, 복잡하면 하드웨어도 필요
	- <img src="https://www.dropbox.com/s/adqh8tvquc9y6wi/Screenshot%202019-06-02%2021.02.22.png?raw=1">

---


### 목적 지향적 모델링 예시
- 병사들의 사과 먹기 예측
	- 목적이 무엇인가? 
		- 1) 먹고 남은 사과 수
		- 2) 사과 먹는데 소요한 시간
		- 3) 장병 1인당 1년에 먹은 사과의 총 수
		- 각각에 대해서 모델, 변수가 달라짐  
- 1) 먹고 남은 사과 수
	- 모델 설계 
		- <img src="https://www.dropbox.com/s/s1a3g3m5sc2esh6/Screenshot%202019-06-02%2021.51.05.png?raw=1">
	- 모델 구현
		- <img src="https://www.dropbox.com/s/pcfwg23qx8fmmuw/Screenshot%202019-06-02%2021.56.22.png?raw=1">
	- 시뮬레이터 검증(Verificaiton)
		- 검증 표를 만들고, 모델의 출력과 계산기를 비교
		- <img src="https://www.dropbox.com/s/1yhh2qrn6hti17k/Screenshot%202019-06-02%2021.57.28.png?raw=1">
	- 모델 실증(Validation)
		- 시뮬레이터와 실제 값의 차이가 왜 나는지? 확인
		- <img src="https://www.dropbox.com/s/y95kiokdqbzty14/Screenshot%202019-06-02%2021.59.15.png?raw=1">

- 2) 사과 먹는데 소요한 시간
	- 모델 설계
		- <img src="https://www.dropbox.com/s/23q9wmyqq8pvrub/Screenshot%202019-06-02%2022.01.14.png?raw=1">
	- 모델의 실증(Validation)
		- 시뮬레이터와 실제 값의 차이가 왜 나는지? 확인
		- <img src="https://www.dropbox.com/s/hatpxvdjx0seyux/Screenshot%202019-06-02%2022.02.33.png?raw=1">
	- 모델의 조정(Calibration)
		- 모델 논리를 개조해서 다시 시뮬레이션하는 과정 => 다시 구현 => 실증
		- <img src="https://www.dropbox.com/s/ss3j1u5vg1i4zws/Screenshot%202019-06-02%2022.03.07.png?raw=1">


  
### Reference
- KAIST 김탁곤 교수님의 [시스템 모델링 시뮬레이션 입문](https://kooc.kaist.ac.kr/isms1) 	
