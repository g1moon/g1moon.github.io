---
layout: post
title:  "영상 이해를 위한 최적화(Optimization) 기법"
subtitle: "ntroduction to Optimization in Computer Vision"
categories: data
tags: optimization
comments: true
---

- Computer Vision에서 사용하는 최적화(Optimization) 기법에 대해 작성한 글입니다


---


### Introduction to optimization
- Optimization
	- 여러 가능한 선택 중 우리가 세운 기준에 입각해 제일 좋은 것을 선택하는 것
	- 비용 함수에 입각해 비용을 최소화시키는 값을 추출
	- Alternative : 대상이 되는 각각의 집합
	- Feasible Set : Alternative의 집합
	- Mathematical programming = Energy minimization (in image understanding)
- Optimization Problem
	- <img src="https://www.dropbox.com/s/kczcaqdx9gv9d7y/Screenshot%202019-06-10%2021.08.23.png?raw=1">
	- f(x)의 최소값이 뭔지보다 최소화하는 x값을 알아내는 것이 중요함


### History
- Linear Algebra
	- 선형 대수
	- Optimization을 잘하기 위해서 고급의 선형 대수 사용
	- 1674년부터 사용
- Optimization as a theoretical tool
	- Gauss가 Least squares problem을 풀었음
		- 가우스 소거법으로 행렬의 역변환을 구함
	- Lagrange
		- 라그랑쥬 기법을 사용해 듀얼 폼(duality)을 만듬
		- nonlinear한 비용 함수에 제한식이 많이 있을 경우, 듀얼 함수로 바꿔서 사용
- Advent of numerical linear algebra
	- 컴퓨터를 이용하거나 수학적 근사를 통해 품
	- Linear Programming in 40s for logistical problems
	- 금융에선 risk를 2차식으로 모델링, 1차식은 기대되는 negative return을 모델링
	- Linear Algebra를 계속 발전시킴
	- 60-70년대에 nonlinear한 문제에 집중
	- 소련에선 컴퓨터가 부족해 이론적으로 돌아가서 무엇이 Optimization 문제를 쉽다 / 어렵다로 구분할 수 있게 하는가?
		- Convexity
		- Convex하면 global minima를 항상 보장해줌 
	- 통계학, 기계공학, 머신러닝 등에서 사용됨
	- Convex opt solvers : CVX, YALMIP

### Classification of optimization techniques in Computer Vision
- <img src="https://www.dropbox.com/s/wep7wcias342cji/Screenshot%202019-06-10%2021.39.23.png?raw=1">
- 어떻게 분류할지는 관점에 따라 다양하게 분류할 수 있음
	- feasible set의 성질에 따라 다름
	- constrain / unconstrain
	- 목적 함수가 linear / nonlinear
- solusion set S의 성질에 의해 분류할 경우 대표적으로 아래처럼 나뉨
	- continous optimization
	- discrete optimization
	- combinatorial optimization
	- variational optimization

- Discrete vs Continuous
	- 3차원의 점을 둘러싼 볼에 다른 점이 없다면 그것을 isolated되었다고 표현함 => discrete set
	- 점을 둘러싼 볼을 아무리 작게 만들어도 항상 또다른 점이 속해있으면 붙어있다고(accumulated) 표현 => continuous
	- 어떤 셋은 discrete + continuous 일 수 있음
- Continuous optimization problems
	- <img src="https://www.dropbox.com/s/2oivkqghs87t5j4/Screenshot%202019-06-10%2021.44.39.png?raw=1">
- Discrete optimization problems
	- <img src="https://www.dropbox.com/s/wcg7rrluj1wsanw/Screenshot%202019-06-10%2021.45.38.png?raw=1">
	- Integer programming : LP relaxtion, discrete한 문제를 continuous optimization으로 생각하고, minimizer를 구함
- Combinatorial Optimization : 숫자를 다룬다기보다 Object라는 객체를 다룸. 객체가 따로 따로 isolate. 따로 빼서 설명하기도 함
	- Graph cut
	- 3개의 픽셀로 이루어진 영상, 픽셀의 칼라값이 있을 때 이걸 전경과 배경 부분으로 나누라는 문제가 주어진다면?
	- 검정색/파란색이 같은 픽셀일 확률이 크다
	- 2^3=8개의 조합 중 내가 정한 비용함수를 최소화하는 것을 선택
	- <img src="https://www.dropbox.com/s/m0fg89tsxfzsgbv/Screenshot%202019-06-10%2021.53.25.png?raw=1">
	- <img src="https://www.dropbox.com/s/2xsj5yj7uranix9/Screenshot%202019-06-10%2021.56.00.png?raw=1">
	- 배경 일부분에 마킹하고 combination
	- <img src="https://www.dropbox.com/s/5b6grp4adlnqhsy/Screenshot%202019-06-10%2021.57.58.png?raw=1"> 
	- <img src="https://www.dropbox.com/s/rjpisrusm0fc00c/Screenshot%202019-06-10%2022.00.09.png?raw=1">
- Variational Optimization
	- 변분 최적화
	- 기존에 continuous와 유사하면서 다른 점은 결과로 나오는 솔루션이 하나의 정수, 벡터 등으로 표시되는데 이 방법은 솔루션이 함수(function)로 나타남
	- Denoising에서 신호 함수, 객체 분할 전경 객체의 boundary를 묘사하는 커브 함수 등을 구할 수 있음
	- 함수로 답이 나와서 infinite-dimensional optimization problem
	- Calculus of Variation을 사용해 주어진 함수의 함수를 최적화
	- functional is a function of another function, 주어진 함수에 함수를 담고 있음
	- functional은 real 값을 할당해야 함
	- continuous partial derivatives
	- <img src="https://www.dropbox.com/s/fu2q3lvr6716l7b/Screenshot%202019-06-10%2022.08.03.png?raw=1">
	- Calculus of Variations
		- 변수를 다루는 방법에서 나아가 함수를 다루는 영역
		- 최소화되는 길은?
		- <img src="https://www.dropbox.com/s/zrcj9etp2f79d75/Screenshot%202019-06-10%2022.10.37.png?raw=1"> 
		- Euler-Lagrange Equation으로 Equation을 바꿔줌
		- <img src="https://www.dropbox.com/s/ynj1bi6p5pzv2nm/Screenshot%202019-06-10%2022.12.30.png?raw=1">
	- <img src="https://www.dropbox.com/s/uaj14buy4sdioy2/Screenshot%202019-06-10%2022.13.55.png?raw=1">
	- <img src="https://www.dropbox.com/s/scz9fyvgffolc6z/Screenshot%202019-06-10%2022.17.34.png?raw=1">

- Constraint 유무에 따른 분류
	- Unconstraint optimization
	- Constraint optimization
- Object function의 모양에 따른 분류
	- Linear programming
	- Nonlinear programming  



### Reference
- [Edwith, 영상이해를 위한 최적화 기법](https://www.edwith.org/optimization2017)
