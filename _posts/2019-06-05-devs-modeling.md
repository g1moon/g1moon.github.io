---
layout: post
title:  "DEVS 모델링 방법론 개요"
subtitle: "DEVS 모델링 방법론 개요"
categories: data
tags: simulation
comments: true
---


- DEVS 모델링 방법에 대한 글입니다


---

### DEVS 모델링 방법론 개요
- 이산사건 모델링의 대표적인 수학적 도구인 DEVS 모델링 방법론 개요
- DEVS(Discrete Event Systems Specification) 형식론
	- 1976년 ~ 1984년 Bernard P.zeigler 교수에 의해 고안
	- 이산사건 모델링을 위한 수학적 틀(3S4F)
		- 완전성, 검증 가능, 수정/확장 용이, 효율적 유지 보수
	- 집합론을 이용한 시스템 이론적 모델 표현
		- 연속시간 모델 표현과 호환성(입력, 출력, 상태, 상태천이 함수)
		- 하이브리드 모델링할 때도 잘 사용할 수 있음
	- 모듈화된 모델의 계층적 명세 기법
		- 모델과 인터페이스가 분리된 구조화된 모델링 가능
	- 학술적 연구 및 다양한 분야(국방, 교통, 재난 등)의 실용적 적용
		- 수학적 기호가 많아 처음에 어려울 수 있지만 그래도 유용
- DEVS M&S 개념
	- 모델과 시뮬레이션 알고리즘은 별개로 분리해 구현
		- 모델과 시뮬레이션 알고리즘이 쌍으로 존재함
		- 모델은 시뮬레이션 알고리즘에 정보를 보내고 시뮬레이션 알고리즘은 그에 따라 모델을 제어
	- DEVS 모델 Class
		- 원자 모델(Atomic Model) : 분해를 하지 않는 최소 단위 모델
		- 결합 모델(Coupled Model) : 원자 혹은 결합 모델들의 결합체
	- 시뮬레이션 엔진
		- 원자모델 시뮬레이션 알고리즘 - 원자 모델 해석기
		- 결합모델 시뮬레이션 알고리즘 - 결합 모델 해석기
- DEVS 모델과 시뮬레이터 구조
	- <img src="https://www.dropbox.com/s/glb9m24if4xw8fp/Screenshot%202019-06-05%2021.52.45.png?raw=1">
- 계층적으로 모듈화된 구조적 모델링
	- 모델 설계 : Top-Down 접근법
	- 모델 구현 : Bottom-up 접근법
- 계층적 모델링 : 재사용 + 정보 은닉
	- 계층적 모델 구조 : 그래프 구조
	- 특징
		- 모델 재 사용 측면
			- A와 B가 결합된 AB는 ABC의 부 모델로 재사용
			- AB와 C가 결합된 ABC는 다른 모델의 부 모델로 재 사용
		- 정보 은닉 측면
			- AB의 부 모델인 A와 B는 C로 직접 접근 불가  
	- <img src="https://www.dropbox.com/s/6zld037kgbavd77/Screenshot%202019-06-05%2021.55.40.png?raw=1">
- 모듈화된 모델링 : 명시적인 입/출력 인터페이스
	- 왼쪽처럼 모듈화된 방법! 
	- <img src="https://www.dropbox.com/s/jqjjsuvcapgeu78/Screenshot%202019-06-05%2022.04.05.png?raw=1"> 

### 원자 DEVS 모델링
- 원자 DEVS 모델 틀 (3S4F : 3 집합 + 4 함수)
	- X : 입력 사건 집합
	- Y : 출력 사건 집합
	- Q : 상태 집합
	- $$q_{T}=(q, e)$$ :시간 명세 상태
	- <img src="https://www.dropbox.com/s/batf0rbuqjxfw5k/Screenshot%202019-06-05%2022.06.46.png?raw=1">
- 원자 DEVS 모델의 4 함수 : 모델링 공식의 의미
	- 시간 전진 함수
		- Q : 현재 상태에서, T : 최대 체류 시간을 정의하라 
	- 출력 함수
		- Q : 현재 상태에서, T : T=ta만큼 시간이 경과하면 출력을 발생하고, 경과하지 않으면 출력 없음 
	- 내부 상태천이 함수
		- Q : 현재 상태에서, T=ta만큼 시간이 경과하면 상태천이하라 
	- 외부 상태천이 함수  
		- Q : 현재 상태에서, T<=ta만큼 시간이 경과했는데 입력이 들어오면 상태를 천이하라 
	- <img src="https://www.dropbox.com/s/c2glwzmi7xfihza/Screenshot%202019-06-05%2022.09.34.png?raw=1">
- DEVS 원자 모델의 표현력
	- ta : 사건 사건 사이의 시간 간격(time out)
		- time out을 명세하는 경우
			- 시간 간격 불규칙 : 이산사건 모델, 워 게임, 에이전트 등 의사결정 표현
			- 시간 간격 일정 : 이산시간 모델, 기동 객체, 센서 등 HW 장비 표현
		- time out을 명세하지 않는 경우
			- Finite State Machine(Finite State Automata), 알고리즘, SW 등 표현
	- <img src="https://www.dropbox.com/s/rhnr0fqsvi1xphf/Screenshot%202019-06-05%2022.11.04.png?raw=1">
- DEVS 모델의 그래프적 표현법 - 상태
	- DEVS 모델링의 수학적 표현이 어렵기 때문에 그래프를 표현
	- <img src="https://www.dropbox.com/s/em2qxd7eg5soeoo/Screenshot%202019-06-05%2022.12.46.png?raw=1">
- DEVS 모델의 그래프적 표현법 - 4 함수
	- 실선 : 외부 상태천이
	- 점선 : 내부 상태천이
	- <img src="https://www.dropbox.com/s/h4bl3todnt2269a/Screenshot%202019-06-05%2022.25.28.png?raw=1">
- 원자 DEVS 모델의 종류
	- 출력만 있는 경우, 입력만 있는 경우, 입력과 출력이 있는 경우
	- <img src="https://www.dropbox.com/s/h8slvjrk18k3jpt/Screenshot%202019-06-05%2022.32.47.png?raw=1">
- DEVS 원자 모델링 절차 : 3S4F 정의
	- 입력, 출력, 상태변수의 집합을 구한다
	- ta 정의
		- 상태 집합 Q의 각 원소 q에 대한 최대 체류 시간 정의
	- $$\delta_{ext}$$ 정의
		- 입력 집합 X와 상태 집합 Q의 모든 가능한 조합 (Q x X)에 대해 다음 상태를 정의
	- $$\delta_{int}$$ 정의
		- 상태 집합 Q의 각 원소 q에 대한 다음 상태를 정의
	- $$\lambda$$ 정의
		- 상태 집합 Q의 각 원소 q에 대한 출력 집합 Y의 각 원소 y를 정의
	- <img src="https://www.dropbox.com/s/2v6cyv8pde4mxg2/Screenshot%202019-06-05%2022.37.05.png?raw=1">

### 결합 DEVS 모델링
- 실 체계 컴포넌트들이 조합된 형태를 모델링하는 결합 DEVS 모델링 방법을 이해
- 순서 쌍(Ordered Pair) 정의 및 표현력
	- 순서 쌍
		- (a, b) <=> a와 b가 쌍으로 존재하며 방향은 a->b다
	- 순서 쌍의 수학적 표현 : (P1, P2)
		- 의미 : 두 지점 P1과 P2는 P1에서 P2 방향으로 연결되어 있다
	- <img src="https://www.dropbox.com/s/kfxqqdm1bz0mvzm/Screenshot%202019-06-05%2022.39.33.png?raw=1">
- 관계(relation) : 순서쌍을 이용한 모델 연결 명세
	- 관계(Relation) : R 순서쌍의 부분 집합
		- 1개의 x에 대해 여러 개의 y가 대응 가능(함수는 1:1 대응)
	- 모델 연결 관계(Coupling Relation : CR)  
		- Mi의 출력 Y와 Mj의 입력 X 연결 관계
		- 1:N의 표현이 가능하므로 한 모델의 출력이 여러 모델의 입력으로 연결되는 것을 표현할 수 있음
	- <img src="https://www.dropbox.com/s/dwh15rqz3dfk155/Screenshot%202019-06-05%2022.41.20.png?raw=1">
- 관계를 이용한 명세 예
	- <img src="https://www.dropbox.com/s/8uo23qiavxja279/Screenshot%202019-06-05%2022.43.26.png?raw=1">
- 관계를 이용한 모델 명세 예
	- 외부 세계와 내부 모델의 연결
	- 내부 모델 사이의 연결
	- 일반적 연결은 위 2개 혼합
	- <img src="https://www.dropbox.com/s/c00l05d5n9ysa6c/Screenshot%202019-06-05%2022.44.51.png?raw=1">
- 결합 DEVS 모델
	- <img src="https://www.dropbox.com/s/ckffz6dzozhi43d/Screenshot%202019-06-05%2022.47.32.png?raw=1">
- 결합 DEVS 모델 명세 예
	- <img src="https://www.dropbox.com/s/e9vdnti6uwc0ilw/Screenshot%202019-06-05%2022.50.43.png?raw=1">
- 멱집합 : 동시 발생 이벤트 우선순위 표현
	- 멱 집합(Power Set)
		- A의 멱 집합은 A의 부분 집합들의 집합
		- DEVS 결합 모델에서 멱 집합 응용 : 사건처리 우선 순위 결정
			- 2개 이상의 모델의 내부사건 처리 시각(TO)이 동일
	- <img src="https://www.dropbox.com/s/9u728708gf7eps1/Screenshot%202019-06-05%2022.52.12.png?raw=1">
- DEVS 결합 모델링 절차
	- <img src="https://www.dropbox.com/s/sc9h306c535az01/Screenshot%202019-06-05%2022.58.24.png?raw=1">

### DEVS 모델링 예시
- 모델링 대상 시스템 및 M&S 목적
	- 모델링 대상 시스템 : 외부 침입자 감시 및 경보 시스템
	- M&S 목적 : 다양한 침입 시나리오에 따른 시스템 동작 검증
- 모델 설계 - 시스템 운용 개념
	- <img src="https://www.dropbox.com/s/khghd155wemv31f/Screenshot%202019-06-05%2023.00.33.png?raw=1">
- 모델 설계 - 모델 구조
	- 침입자 모델 / 감시 및 경고 모델
	- <img src="https://www.dropbox.com/s/w0l83ch632vngvq/Screenshot%202019-06-05%2023.01.34.png?raw=1">
- 모델 설계 - 결합 모델
	- <img src="https://www.dropbox.com/s/lx06x7vq3x9ieyo/Screenshot%202019-06-05%2023.03.47.png?raw=1">
- 모델 설계 - 원자 모델
	- <img src="https://www.dropbox.com/s/8sjwp10sefnlh4h/Screenshot%202019-06-05%2023.16.51.png?raw=1">
	- <img src="https://www.dropbox.com/s/so6vgbpv4pv0rnx/Screenshot%202019-06-05%2023.17.17.png?raw=1">
- 전체 시스템 모델 설계 - 결합 모델
	- <img src="https://www.dropbox.com/s/ti3n8wgfff1r6re/Screenshot%202019-06-05%2023.17.53.png?raw=1">
- DEVS 모델 실행 결과
	- <img src="https://www.dropbox.com/s/rnrocksqx9y3xzt/Screenshot%202019-06-05%2023.18.51.png?raw=1">
      
     



### Reference
- KAIST 김탁곤 교수님의 [시스템 모델링 시뮬레이션 입문](https://kooc.kaist.ac.kr/isms1)


 
