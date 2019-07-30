---
layout: post
title:  "시스템 모델링 방법"
subtitle: "시스템 모델링 방법"
categories: data
tags: simulation
comments: true
---


- 시스템 모델링 방법에 대한 글입니다


---


### 집합 기호를 사용한 모델 표현 방법
- 시스템 모델링과 집합 이론
	- 집합 : 유사한 특성을 갖는 서로 다른 원소들의 모임
	- 모델 표현에 사용된 집합 연산들
		- 원소, 부분집합, 적집합, 멱집합
		- 함수 f: A->B
		- 관계 R ⊆ A X B  
	- 집합이론 기반 시스템 모델링 특징
		- 집합 연산을 사용해 간단 명료하게 시스템 모델 표현
		- 집합 연산의 결과는 집합
			- 상태 변수의 구조화 (Q = Q1 X Q2)
			- 시스템의 모듈화 및 계층화 (DEVS 형식론)
- 집합을 사용한 상태 변수, 입력, 출력, 컴포넌트 표현
	- X : 입력 사건 집합
	- Y : 출력 사건 집합
	- Q : 상태 변수 집합
	- 계층적 표현
	- <img src="https://www.dropbox.com/s/db2m6v6slu761yj/Screenshot%202019-06-03%2001.16.55.png?raw=1">  
- 적 집합을 사용한 상태변수의 구조화
	- 적 집합의 정의
		- A는 B, C의 집합 (Combination)
	- <img src="https://www.dropbox.com/s/ji3wo65491vle5p/Screenshot%202019-06-03%2001.18.07.png?raw=1">
- 함수를 사용한 상태 천이(상태변수 값의 갱신) 표현
	- 시스템에서 관찰된 상태천이 과정을 델타로 표현
	- <img src="https://www.dropbox.com/s/mvqa6cneexjtfff/Screenshot%202019-06-03%2001.21.46.png?raw=1"> 
- 함수, 알고리즘, 테이블, DB : 동일한 표현력
	- <img src="https://www.dropbox.com/s/qpddsxtqhix67xm/Screenshot%202019-06-03%2001.25.49.png?raw=1">
- 집합 기호로 표현된 모델의 다른 표현법
	- 다이어그램 기반 모델 표현시 f(.)을 씀
	- <img src="https://www.dropbox.com/s/7wab7g2citgnva4/Screenshot%202019-06-03%2001.27.32.png?raw=1">
- 집합을 사용한 모델 표현 예
	- 수학 기호, 그래프 모두 동일한 표현력을 가짐
	- <img src="https://www.dropbox.com/s/0irja0lvf9zwq4x/Screenshot%202019-06-03%2001.28.29.png?raw=1"> 

---

### 연속시간 및 이산사건 모델링
- 모델 상태 변환이 시간기반인 연속시간 모델과 사건기반인 이산사건 모델의 특징을 이해하기
- 집합 이론적 시스템 모델링
	- 주어진 시스템이 입력 X, 상태 변수 Q, 출력 Y으로 구성될 경우
	- 모델링을 통해 3 집합 X, Y, Q의 제약조건 및 이들 사이의 제약조건(관계) 표시
	- 3개 집합 각각의 제약 조건을 표시하고, 3 집합 사이의 제약 조건(관계)를 표시
	- <img src="https://www.dropbox.com/s/3wzekbiyimk1j4b/Screenshot%202019-06-04%2000.22.54.png?raw=1">
- 연속시간 및 이산사건 M&S 도구/환경
	- 연속 시스템과 이산 시스템일 경우에 따라 도구가 다름
	- 연속 시스템은 주로 H/W 객체 기능 모델링(장비, 자산 등)
		- 제약 조건은 미분 방정식
	- 이산사건 시스템은 S/W 객체 프로세스 모델링(운용, 서비스, 의사결정 등) 
		- 제약 조건은 미분 방정식을 이산으로 변경
	- 이산 사건은 완전한 범용 라이브러리 구축은 힘들고 개별로 만듬
	- <img src="https://www.dropbox.com/s/o1a1i2d8hhdbe67/Screenshot%202019-06-04%2000.25.25.png?raw=1"> 
- 시스템 모델 = 상태 방정식 + 출력 방정식
	- 입력이 있을 때, 없을 때마다 상태 방정식 + 출력 방정식이 존재
	- <img src="https://www.dropbox.com/s/abj4lau1nt7aa0z/Screenshot%202019-06-04%2000.27.24.png?raw=1">
- 연속시간 모델링과 이산사건 모델링 예
	- 자동차가 움직임
	- 모델링 목적에 따라 연속시간, 이산사건으로 모델링할 때가 존재
	- 모델링 목적1
		- 속도 v(t)로 달리는 자동차의 x 방향 이동거리 s(t) 구하기
		- 모델링 : 속도 v(t)와 거리 s(t) 사이의 관계식 -> 연속시간 모델
	- 모델링 목적2
		- 신호등의 적,녹, 황색을 정해진 시간 간격으로 유지시키는 신호 체계
		- 모델링 : 정해진 시간 간격으로 신호등 색깔 변경을 주기적으로 반복하는 신호등 모델 -> 이산사건 모델
	- <img src="https://www.dropbox.com/s/bekds0vh451nmsz/Screenshot%202019-06-04%2000.29.25.png?raw=1">
- 연속시간 모델 - 자동차 이동 거리
	- 시뮬레이션 => 모델 실행 => 미분 방정식 풀기
	- 예시는 정말 간단한 공식으로 가능하지만, 현실은 공식보다 수치 해석으로 진행
	- <img src="https://www.dropbox.com/s/okbl8wlnq7zs2uc/Screenshot%202019-06-04%2000.34.50.png?raw=1">  	 
- 이산사건 모델 - 신호등 체계
	- T1 시간에 적색, T2 동안 황색, T3 동안 녹색
	- (T1+T2+T3) 주기로 반복
	- 타임아웃 이벤트를 받아서 상태가 바뀜!!
	- 여러 모델이 있으면 모든 모델의 시간을 관리하고 시간이 작은 순서로 모델을 실행하고, 모델의 결과가 또다른 모델의 입력이 됨
	- <img src="https://www.dropbox.com/s/hypqtrb6rjgl8zw/Screenshot%202019-06-04%2000.37.32.png?raw=1">


--- 

### 이산사건 정의 및 사건 중심 모델링 방법
- 이산사건을 정의하고 이산사건 모델링 방법을 소개
- 이산사건과 이산사건 시뮬레이션이란?
	- 사건 : 메세지 혹은 명령(시스템 전체에 유한 개 사건 존재)
	- 이산 : 사건 발생 시점에 대한 패턴
	- 모델 구현시 프로그램에서 사용되는 함수 이름
		- 사건 발생 = 함수 호출
		- 사건 처리 = 함수 실행
	- 이산사건 시스템 모델
		- 대기, 기동, 탐지
	- 이산
		- 사건과 사건 사이의 간격은 이산(불연속)적이며 사건 발생 시각은 랜덤함 (대기-기동-탐지-대기 등 랜덤하게)    
	- 이산사건 시뮬레이션
		- 임의의 시각에 이산적으로 발생되는 사건들을 처리하기 위한 해당 함수들의 호출
	- <img src="https://www.dropbox.com/s/41qtkvjwpj16z5g/Screenshot%202019-06-04%2000.45.08.png?raw=1">
- 이산사건 시스템 모델링 접근 방법
	- 월드 뷰 모델링(프로그래밍적 접근)
		- 시스템 동작을 프로그래밍 관점에서 바라보며 모델 표현 
		- 사람에 따라 프로그래밍 관점이 다르고 구현 방법도 다름
		- 사건 중심(Event-oriented) 관점
			- 시스템 내부의 사건 발생을 추적해 시스템을 표현 
		- 프로세스 중심(Process-oriented) 관점
			- 시스템 내부의 관심 대상 객체가 생성되어 소멸될 때까지의 과정을 사건 열로 표현
		- 객체 중심(Object-oriented) 관점
			- 시스템을 구성하는 각 객체들의 특성과 이들 사이의 상호작용으로 시스템 표현  
	- 시스템 이론적 모델링(정해진 모델링 틀)
		- 시스템 이론에서 제시한 모델링 틀(입력, 출력, 상태변수)에 의해 모델 표현
		- 모든 사람이 동일한 틀에 모델을 표현
		- DEVS(Discrete Event Systems Specification) 형식론
			- 시스템을 구성하는 각 객체들을 계층적으로 나누어서 모듈화
			- 객체 중심과 관점은 동일하지만 객체를 표현할 때 시스템 이론적 틀을 사용 => 객체 중심(지향적)으로 시스템 이론적 모델 표현
	- <img src="https://www.dropbox.com/s/x6f1pqbosx4xmp9/Screenshot%202019-06-04%2000.55.42.png?raw=1">
- 이산사건 모델링: 월드 뷰 vs 시스템 이론
	- 시스템 이론적 수학적 모델로 완전성, 검증성, 통신 수단 등의 장점을 가짐
	- <img src="https://www.dropbox.com/s/6rb6m7uykr587ax/Screenshot%202019-06-04%2001.13.37.png?raw=1">
- 대표적 월드 뷰 : Event-oriented 모델링 개념
	- 대상 시스템을 관찰하고 시스템 내부에서 발생하는 이벤트와 시간 간격이 그림과 같을 때, 사건마다 사건 처리 함수를 구현
	- 사건 처리 함수 Event1 => T1 시간 후에 호출 => 사건 처리 함수 Event5
	- 사건 처리의 3단계
		- 1) 호출된 사건 처리
		- 2) 사건 처리 소요 시간 계산 
		- 3) 다른 사건처리함수들 호출
	- <img src="https://www.dropbox.com/s/aklarpacywjevj0/Screenshot%202019-06-04%2001.15.50.png?raw=1">
- Event-oriented 모델의 프로그램적 관점
	- 대상 시스템에 필요한 모든 변수를 전역화 해서 변수 공간을 가진 후, 그 연산을 이벤트 함수로 생성
	- <img src="https://www.dropbox.com/s/4e4bx9h8jmkm2lz/Screenshot%202019-06-04%2001.16.36.png?raw=1">
- Event-oriented 모델 실행 => 시뮬레이션 방법
	- 시뮬레이션 엔진
	- Event List : 함수와 호출 시간을 시간 순으로 관리 
	- 이벤트를 등록할 때 Schedule을 추가한다고 함 
	- <img src="https://www.dropbox.com/s/4iemavefwwxr1dv/Screenshot%202019-06-04%2001.18.25.png?raw=1">
- 이산사건 모델 구현 예 - 신호등 모델
	- schedule(event, delay time)
	- main()에서 schedule(Event_적색켜기, 0)
	- <img src="https://www.dropbox.com/s/y7sy4r3cdbhpcd8/Screenshot%202019-06-04%2001.20.41.png?raw=1">

---

### 시스템 이론적 이산사건 모델링
- 이산사건 모델링 방법 중 수학적 형식론을 이용한 시스템 이론적 모델링 틀을 소개
- 시스템 이론적 이산사건 모델 표현
	- 시스템을 입력(X), 출력(Y), 상태 변수(Q)의 집합으로 표현
	- 세 집합의 제약 조건 및 이들 사이의 제약조건 관계를 정의하는 것을 시스템 이론적 모델링이라 함
	- Q를 통한 간접 관계식 정의(직접 정의 안함)
	- <img src="https://www.dropbox.com/s/06zcdq5yp2pnbo0/Screenshot%202019-06-04%2001.28.43.png?raw=1">
- 이산사건 모델의 상태천이 : 내,외부 사건 발생시 처리
	- 이산 사건은 현재 상태에서 최대 체류 시간 설정
	- Time out 발생하면 체류시간 만료(내부 시간)
	- 외부 사건 처리
		- 외부 사건(X)이 발생할 때마다 각 사건별 상태 변수 갱신
	- 내부 상태 처리
		- 내부 사건(Time Out)이 발생할 때마다 상태 변수 갱신 
	- X와 TO 발생 시각은 서로 독립적
	- <img src="https://www.dropbox.com/s/mnp617x7ljv8wzz/Screenshot%202019-06-04%2001.32.01.png?raw=1">
- 이산사건 모델의 출력 사건 : 내부 사건 발생시
	- 내부 사건이 발생할 때마다 출력 함수 실행
	- <img src="https://www.dropbox.com/s/ju0stz4unwx5t5i/Screenshot%202019-06-04%2001.38.13.png?raw=1">
- 내부 상태천이와 최대 체류시간
	- 실제 상황 : 걷기 50분, 10분 휴식, 다시 30분 뛴다
	- 이산사건 모델링
		- 외부 입력(X) 없이 정해진 스케쥴(최대 체류시간)만 실행하는 모델
	- <img src="https://www.dropbox.com/s/lgy6bmx302xjftv/Screenshot%202019-06-04%2001.43.39.png?raw=1">
- 함수를 사용한 최대 체류시간 명세
	- 함수를 사용해 각 상태별로 값을 명세. mt
	- mt(걷기) = 50, mt(휴식)=10, mt(뛰기)=10
	- 평균이 몇이고 분산이 +-5고 정규분포다 이런 식으로 체류시간을 정의함
	- <img src="https://www.dropbox.com/s/lf9bbjgpzv7k2k4/Screenshot%202019-06-04%2001.44.42.png?raw=1"> 
- 시간 명세 상태(Timed State) 정의 및 시간 명세 상태천이
	- 걷기 50분 휴식 10분 뛰기 30분으로 하면 상태와 상태에서 체류 시간을 묶어서 정의
	- 걷기를 얼마나 했다를 표시할 수 있음
	- <img src="https://www.dropbox.com/s/xe53xtz8jkiagnw/Screenshot%202019-06-04%2001.47.48.png?raw=1">
- 시스템 이론적 이산사건 모델: 상태천이 + 출력 발생 명세
	- <img src="https://www.dropbox.com/s/e6sfld81dkq40gb/Screenshot%202019-06-04%2001.48.38.png?raw=1">
- 이산사건 모델의 형태와 상태천이/출력
	- <img src="https://www.dropbox.com/s/8hcm0bgj9gzgru6/Screenshot%202019-06-04%2001.49.09.png?raw=1">  




### Reference
- KAIST 김탁곤 교수님의 [시스템 모델링 시뮬레이션 입문](https://kooc.kaist.ac.kr/isms1)


 
