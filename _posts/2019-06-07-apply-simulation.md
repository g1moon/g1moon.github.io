---
layout: post
title:  "시뮬레이션 모델 개발 방법론 및 M&S 활용 방법"
subtitle: "시뮬레이션 모델 개발 방법론 및 M&S 활용 방법"
categories: data
tags: simulation
comments: true
---


- 시뮬레이션 모델 개발 방법론 및 M&S 활용 방법에 대한 글입니다


---


### M&S에서 통계의 역할 및 실험 설계법
- M&S 프로세스에서 통계학의 필요성
	- 목적에 부합된 데이터 수집 / 실증 / 결과 분석할 때 통계가 필요
	- 모델 내부 변수들이 비 확정적 확률 변수
	- 입력 값이 확률 변수면 출력 값도 확률 변수
	- 실증 과정은 난수를 사용한 통계적 가설검증
		- 데이터 실증(Validataion)
		- 모델 실증(Validation)
	- 결과 분석은 출력 난수 값들의 통계적 처리
	- <img src="https://www.dropbox.com/s/8ljhfwnfzc1yfze/Screenshot%202019-06-06%2011.51.17.png?raw=1">
- 통계 이론의 M&S 적용 개념
	- 모집단에거 표본을 추출해 모집단의 통계값 예측
	- M&S에선 실제 시스템을 모집단이라 생각하고 모델은 표본이라 생각할 수 있음
	- <img src="https://www.dropbox.com/s/abtl4axdop5go8w/Screenshot%202019-06-06%2012.02.36.png?raw=1">
- 통계 용어와 M&S 용어의 유추적 비교
	- <img src="https://www.dropbox.com/s/ju4d7zc6dgnfoti/Screenshot%202019-06-06%2011.57.19.png?raw=1">
- 실험 틀(Experimental Frame) 개념
	- 실험 틀 모델
		- 입력 데이터 발생기 : 실험 시나리오 및 실험 설계 데이터를 시스템 모델에 입력
		- 시뮬레이션 제어기 : 논리적 오류 식별 및 실행 제어(파라미터 a는 반드시 양수가 되야한다! a가 음수가 되면 중단)
		- 출력 데이터 수집기 : 통계처리/분석에 필요한 데이터
	- 시스템 모델과 실험 틀 모델이 완전 분리되어야 함
		- 시뮬레이션 실험의 효율성
			- 시스템 모델과 실험 틀 모델의 조합으로 다양한 실험 수행
			- 모델 1번 => 실험 틀 여러개
			- 실험 틀 고정 => 모델 여러개 
		- 모델 수정 / 유지 보수의 효율성  
			- 시스템 모델과 실험 틀 모델의 독립적 수정/유지보수
	- <img src="https://www.dropbox.com/s/7is0eblyh4xfgkk/Screenshot%202019-06-06%2012.01.39.png?raw=1"> 
- 시뮬레이션 실험 설계
	- 굉장히 중요함
	- 모든 입력 변수 값 조합들의 집합을 X로 넣음
	- 그러나 오래 걸릴 수 있어서 집합의 일부만 표본으로 보고 추세를 볼 수 있음 => 최소의 실험
	- <img src="https://www.dropbox.com/s/ad868j8aucs9fy7/Screenshot%202019-06-06%2012.06.41.png?raw=1">
- 실험 설계 순서
	- 1) 문제 정의 및 실험 목적 설정
	- 2) 측정 지수 정의(Y) 및 측정 대상 변수(X) 결정
	- 3) 대상 변수 값들의 변경 범위 설정
	- 4) 변수 값 선택 설계
	- 5) 실험 및 데이터 수집
	- 6) 데이터 분석
	- 7) 결론 및 추천
	- <img src="https://www.dropbox.com/s/9halai3ux91llof/Screenshot%202019-06-06%2012.07.44.png?raw=1">
- 고전적 실험 설계의 대표적 예시
	- Full Factorial Design, FFD : 모든 가능한 입력 변수 조합 사용
		- 문제점 : 많은 실험 횟수로 비용 증대
	- Fractional Factorial Design : 모든 가능한 조합 중 일부 조합 사용
		- 문제점 : 어떤 조합이 실험 설계 목표 달성이 가능한지 판단이 난해함
	- 더 궁금하면 실험 설계법 학습을 추천
	- <img src="https://www.dropbox.com/s/ninc96bxv3pjwpq/Screenshot%202019-06-06%2012.09.32.png?raw=1">
- 복제 시뮬레이션을 통한 표본 추출 방법
	- 시뮬레이션 R번 반복(복제 시뮬레이션)
	- <img src="https://www.dropbox.com/s/moik8dwd2wwiyp9/Screenshot%202019-06-06%2012.10.47.png?raw=1">
- R개의 출력 표본 추출을 위한 복제/반복 실험 방법
	- 복제 : 생성된 데이터의 순서는 다르지만 통계적 특성이 같은 순차적 데이터
	- <img src="https://www.dropbox.com/s/jif6u3mc2o6195q/Screenshot%202019-06-06%2012.11.56.png?raw=1">     
- R개의 출력 표본 추출을 위한 복제/반복 실험 방법
	- 크기를 1/R 씩 R개로 분할
	- <img src="https://www.dropbox.com/s/6no3xpzmwcgnw9t/Screenshot%202019-06-06%2012.14.28.png?raw=1"> 

---

### M&S 체계 개발 방법, 환경 및 적용 사례
- M&S 체계 개발 누가, 어떻게 하나?
	- 일기예보 예시
	- <img src="https://www.dropbox.com/s/hpd3ph09l33pmxa/Screenshot%202019-06-06%2019.26.34.png?raw=1">
	- 국방 M&S 체계 예시
	- <img src="https://www.dropbox.com/s/1755l9wwkkq58wd/Screenshot%202019-06-06%2019.27.04.png?raw=1">
- 국방 M&S 체계
	- 군사학 + M&S 기술 + ICT 기술
	- <img src="https://www.dropbox.com/s/opn5do5tm3tj3oq/Screenshot%202019-06-06%2019.29.22.png?raw=1">
- 시뮬레이션 시스템
	- 도메인 지식 + M&S 기술 + ICT 기술
	- <img src="https://www.dropbox.com/s/fh3p79ogknnnxiq/Screenshot%202019-06-06%2019.31.50.png?raw=1">
- SW 공학 측면에서 본 이산사건 M&S 시스템 개발
	- <img src="https://www.dropbox.com/s/nukdlqb4i89gddv/Screenshot%202019-06-06%2019.33.25.png?raw=1">
- M&S 체계 연동 개념
	 - 체계 연동 : 체계들 간 데이터 교환 + 시뮬레이션 시각 동기화
	 - <img src="https://www.dropbox.com/s/aiwd81byc2nsk9f/Screenshot%202019-06-06%2019.38.46.png?raw=1">
 - KAST SMSLab 프로젝트에서 사용하는 다양한 도구들
 	- <img src="https://www.dropbox.com/s/34xt8lcwkx9cw5v/Screenshot%202019-06-06%2019.39.24.png?raw=1">
- 개발 사례 : 기만기 효과도 분석 M&S 체계
	- 목적 : 효과도 분석을 통한 기만기 제원 산출 및 운용 전술 개발
		- 기만기 : 전자전 무기체계 (음파 발생기)
	- <img src="https://www.dropbox.com/s/hwzoscs3hk0q8ft/Screenshot%202019-06-06%2019.41.06.png?raw=1">
- 효과도(회피율) 분석 M&S 체계 상위 레벨 설계
	- 몬테 카를로 시뮬레이션 사용
	- <img src="https://www.dropbox.com/s/o25cn5syxbpdekb/Screenshot%202019-06-06%2019.42.02.png?raw=1">
- 연동 시뮬레이션 설계
	- 연속 시간 모델 + 이산 사건 모델 연동
	- 수상함은 이산 사건 모델
	- 아래 기만기와 어뢰는 연속 시간 모델
	- <img src="https://www.dropbox.com/s/fptv7ie9nlemwfg/Screenshot%202019-06-06%2019.42.44.png?raw=1">
- 기만기 효과도 분석 시뮬레이션 과정
	- <img src="https://www.dropbox.com/s/a98cyqy1nupaq3q/Screenshot%202019-06-06%2019.43.00.png?raw=1">
- 시뮬레이션 결과
	- 교리 개발 vs 기만기 제원(ROC) 도출
	- <img src="https://www.dropbox.com/s/gkhwo7takztk2ka/Screenshot%202019-06-06%2019.43.41.png?raw=1">

---

### M&S 활용 방안
- M&S 기법 적용
	- 시스템 과학의 3가지 문제
	- 분석 문제
		- 기능 분석 / 검증
		- 성능(효과도) 분석 / 예측
		- 고장 진단 / 수명 예측 등
	- 설계 문제
		- 설계 명세 및 검증
		- 시뮬레이션 기반 획득 등
	- 최적화 문제
		- 정적 최적화(파라미터 최적화)
		- 동적 최적화(강화학습)
- 시스템 수명 주기(Life Cycle)에서 M&S 적용 방안
	- 전체 Life Cycle에 적용 가능 
	- <img src="https://www.dropbox.com/s/3lowxnnk551b3ob/Screenshot%202019-06-06%2019.53.39.png?raw=1"> 
- 시뮬레이션 기반 획득(SBA : Simulation Based Acquisition)
	- SVA = SE의 V 프로세스 + M&S 기술 + 제조/생산 기술
	- <img src="https://www.dropbox.com/s/ow19rh4utzndarg/Screenshot%202019-06-06%2019.56.34.png?raw=1">
- 시뮬레이션 기반 최적화 문제
	- 현실 세계를 수식 형태로 표현 불가능할 경우 시뮬레이션 기반 최적화 문제로 품
	- 일반 최적화 문제는 목적 함수와 제약 조건이 닫힌 형태의 수식으로 표현
	- <img src="https://www.dropbox.com/s/62cysvbwg4ktv4t/Screenshot%202019-06-06%2019.58.34.png?raw=1">  
- Stochastic 시뮬레이션 기반 최적화 : 정적 vs 동적
	- 정적 최적화 문제
		- 메타 휴리스틱 알고리즘
		- 최적화된 x값은 시스템 상태 S와 무관
	- 동적 최적화 
		- 강화 학습(Q 알고리즘)
		- 최적화된 a 값은 시스템 상태 S의 함수
	- <img src="https://www.dropbox.com/s/u7mhiwkcsch6r0m/Screenshot%202019-06-06%2020.00.05.png?raw=1">
- 국방 분야에서 M&S 활용
	- <img src="https://www.dropbox.com/s/sg41lgn9kuzp5b8/Screenshot%202019-06-06%2020.00.56.png?raw=1">
- 이산사건 M&S 시스템 활용 : Simulation As Service
	- 대상 이산사건 시스템
	- 이런 시스템 중 M&S 시스템 구축
	- 도메인 지식을 구현
	- <img src="https://www.dropbox.com/s/dknklod2x16t7d9/Screenshot%202019-06-06%2020.02.41.png?raw=1">

---

### 동적 시스템의 데이터 모델 및 시뮬레이션 모델
- 머신러닝을 이용한 모델과 시뮬레이션 모델의 차이점을 이해하기
- 동적 시스템 모델링 접근법 : 기계학습 vs 시스템 모델링
	- 동적 시스템 : 시스템의 상태가 계속 달라짐
	- 기계학습은 데이터를 통해 모델을 생성
	- 시스템 과학은 물리 법칙을 사람이 시뮬레이션 모델로 변환 
	- 목적은 다양한 의문 사항을 풀기 위함(공통점)
		- 예측
	- 입력 시나리오를 넣어 가상 실험을 파악
	- <img src="https://www.dropbox.com/s/trihsx6k3hepg30/Screenshot%202019-06-06%2020.05.48.png?raw=1">
- 기계학습 기반 데이터 모델의 구조적 제한점
	- 예측 결과가 유효할 조건
		- 학습할 때와 예측할 때의 시스템 환경/조건이 동일해야 함
		- <img src="https://www.dropbox.com/s/zup93x7wiiwmhil/Screenshot%202019-06-06%2020.08.25.png?raw=1">
- 시스템 동작 공간 : 데이터 모델 vs 시스템 모델
	- 직사각형이 되는 것은 데이터 모델에선 불가능
	- <img src="https://www.dropbox.com/s/xy4p8btfc3bpjyf/Screenshot%202019-06-06%2020.11.11.png?raw=1">        
- 동적 시스템 예측 및 최적화 : 데이터 모델 vs 시스템 모델
	- <img src="https://www.dropbox.com/s/1v0m6rzwxqh9bcw/Screenshot%202019-06-06%2020.12.30.png?raw=1">
- 동적 시스템 분석 능력 : 데이터 모델 vs 시스템 모델
	- <img src="https://www.dropbox.com/s/39lu1s1nmep35of/Screenshot%202019-06-06%2020.13.33.png?raw=1">
- 귀납적 접근과 연역적 접근의 만남
	- AI + M&S 융합
	- 귀납적 방법 : 데이터를 가지고 출발
	- 연역적 방법 : 내부 지식으로 접근
	- 어떻게 융합할지가 중요함
	- <img src="https://www.dropbox.com/s/osruxtl5thpxjq3/Screenshot%202019-06-06%2020.14.46.png?raw=1">  


---

### 추가적으로 볼 자료
- 캠브릿지 대학의 [Machine Learning for Materials Science](file:///Users/kyle/Downloads/Rouet-Leduc-2017-PhD.pdf)
- [Combining Simulation with Machine Learning to Build Accident Models](https://www-users.cs.york.ac.uk/~rda/sasemas_06_paper.pdf)
- [Data modeling versus simulation
modeling in the big data era: case
study of a greenhouse control system](https://journals.sagepub.com/doi/pdf/10.1177/0037549717692866)
 

### Reference
- KAIST 김탁곤 교수님의 [시스템 모델링 시뮬레이션 입문](https://kooc.kaist.ac.kr/isms1)


 
