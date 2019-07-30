---
layout: post
title:  "Simulated Annealing 개념과 Python 구현"
subtitle: "Simulated Annealing 개념과 Python 구현"
categories: data
tags: optimization
comments: true
---

- Simulated Annealing에 대한 설명과 파이썬 구현 코드를 작성한 글입니다

---

## Simulated Annealing
- Kirkpatrick이 1983년에 만듬
- 뜨거운 욕조에서 재료의 냉각을 시뮬레이션하는 알고리즘에 기반
- 더 전통적인 방법의 변형
	-  Local(neighborhood) search
- 확률론적 메타휴리스틱 방법
- Annealing
	- 내부 강도를 제거하기 위해 금속이나 유리를 가열하고 천천히 냉각시키는 방법
	- 금속재료를 가열한 다음 조금씩 냉각해 결정을 성장시켜 그 결함을 줄이는 작업
	- 열에 의해서 원자는 초기의 위치(내부 에너지가 극소점에 머무르는 상태)로부터 멀어져 에너지가 더욱 높은 상태로 추이됨
	- 천천히 냉각함으로써 원자는 초기 상태보다 내부 에너지가 한층 더 극소인 상태를 얻을 가능성이 많아짐
- Cooling하는 방법
	- Temperature를 낮추는 방법
	- Cooling rate  
- Local minima/optima를 벗어나는 컨트롤된 방식으로 오르막 이동(더 나쁜 솔루션)을 허용함
	- 확률로 더 나쁜 움직임을 받아들임
	- random number를 체크한 후, random이 적으면 나쁘게 움직이고 높으면 움직이지 않음 
- <img src="https://www.dropbox.com/s/4w56bige61wyoed/Screenshot%202019-03-26%2023.53.53.png?raw=1">
- <img src="https://www.dropbox.com/s/tldpmm7ds7gqneh/Screenshot%202019-03-27%2000.01.01.png?raw=1">
- 해를 반복해 개선하며 현재의 해 근방에 있는 해를 임의로 찾는데, 그 때 주어진 함수의 값과 전역 인자 T가 영향을 줌
	- 위에서 말한 물리 과정과 비슷한 원리로 T(온도)의 값은 서서히 작아짐
	- 처음엔 T가 크기 때문에 해가 크게 변화하지만, T가 0에 가까워짐에 따라 변화가 줄어듬
	- 처음은 간단하게 비탈을 올라갈 수 있으므로, 지역 최적점에 빠졌을 때 대책을 생각할 필요가 없음
- 아이디어 자체는 모든 분야에 대하여 적용이 가능
	- 임의의 경우의 수가 많은 경우 정해진 조건에서 대용량의 최적화를 찾을 때 유용하게 사용 


### Notations
<img src="https://www.dropbox.com/s/mdbt91lfzeafx4q/Screenshot%202019-03-28%2020.39.11.png?raw=1">

- 빨간색이 Final Solution

<img src="https://www.dropbox.com/s/wxuijdt6m3ig532/Screenshot%202019-03-28%2020.40.42.png?raw=1">

### 방법
- $$T_{0}$$, M, N, alpha, move operator의 종류를 설정
- $$m = 1$$
- Search space $$x_{i}$$에서 random point로 시작
- move operator $$x_{t}$$를 사용해 다른 장소로 이동
- 근처를 둘러보고 그 중 하나로 이동
- 더 나아졌는지 확인 후, 나아졌으면 끝. 아니면 계속
	- 나아지지 않았다면 random number를 취함
	- $$1/(e^{(f(x_{tmp})-f(x_{i})/T_{t})}$$랑 비교
	- 작다면 취하고, 작지 않다면 다른 곳을 찾고 n = n + 1
- N번 진행
- m = m + 1
- $$T_{t+1} = \alpha * T_{t}$$
- 1time에 5~9 스텝을 반복
- $$x_{Final}$$을 찾고 좋은 Solution으로 기록

### Flowchart
<img src="https://www.dropbox.com/s/repgqi2ne3n2nxu/Screenshot%202019-03-28%2021.04.37.png?raw=1">

### Himmelblau 구현
- [Himmelblau function](https://en.wikipedia.org/wiki/Himmelblau%27s_function)
- $$ z = (x^{2} + y - 11)^{2} + (x+y^{2}-7)^{2}$$
- x와 y는 -6 ~ 6
- minimized z = 0.0000
- 1개의 극대값과 4개의 극소값을 가짐


### Himmelblau 코드
```
import numpy as np
import matplotlib.pyplot as plt


x0 = 1 # Initial solution
y0 = -1

k = 0.1
T0 = 1000
M = 300
N = 15
alpha = 0.85


z_int = ((x0**2)+y0-11)**2+(x0+(y0**2)-7)**2

print(f"Initial X is {x0:.3f}")
print(f"Initial Y is {y0:.3f}")
print(f"Initial Z is {z_int:.3f}")
```

- 처음엔 Z가 146으로 Optimal과 거리가 멈

```
import numpy as np
import matplotlib.pyplot as plt


x0 = 1 # Initial solution
y0 = -1

k = 0.1
T0 = 1000
M = 300
N = 15
alpha = 0.85


z_int = ((x0**2)+y0-11)**2+(x0+(y0**2)-7)**2

print(f"Initial X is {x0:.3f}")
print(f"Initial Y is {y0:.3f}")
print(f"Initial Z is {z_int:.3f}")

temp = []
min_z = []

# neighborhood search

for i in range(M):
	for j in range(N):
		xt = 0
		yt = 0
		
		# move operator
		ran_x_1 = np.random.rand()
		ran_x_2 = np.random.rand()
		ran_y_1 = np.random.rand()
		ran_y_2 = np.random.rand()
		
		if ran_x_1 >= 0.5:
			x1 = k*ran_x_2
		else:
			x1 = -k*ran_x_2
			
		if ran_y_1 >= 0.5:
			y1 = k*ran_y_2
		else:
			y1 = -k*ran_y_2
			
		xt = x0 + x1
		yt = y0 + y1
		
		of_new = ((xt**2)+yt-11)**2+(xt+(yt**2)-7)**2
		of_final = ((x0**2)+y0-11)**2+(x0+(y0**2)-7)**2
	
		ran_1 = np.random.rand()
		form = 1/(np.exp((of_new-of_final)/T0))
		
		if of_new <= of_final:
			x0 = xt
			y0 = yt
		elif ran_1 <= form:
			x0 = xt
			y0 = yt				
		else:
			x0 = x0
			y0 = y0
		
	temp = np.append(temp, T0)
	min_z = np.append(min_z, of_final)
	T0 = alpha*T0
	
print(f"X is {x0:.3f}")	
print(f"Y is {y0:.3f}")	
print(f"Final OF is {of_final:.3f}")	

plt.plot(temp, min_z)
plt.title("Z vs Temp", fontsize=20, fornweight='bold')
plt.xlabel("Temp", fontsize=18, fornweight='bold')
plt.ylabel("Z", fontsize=18, fornweight='bold')

plt.xlim(1000, 0)
plt.xticks(np.arrange(min(temp), max(temp), 100), fornweight='bold')
plt.yticks(fontweight='bold')

plt.show()


>>> X is 3.584
>>> Y is -1.848
>>> Final OF is 0.000
```

<img src="https://www.dropbox.com/s/88bwlvqwrgpj5ki/Screenshot%202019-03-29%2020.47.40.png?raw=1">

- x0를 2로 바꾸고, y0을 1로 바꾸고 실행해보기


### Quadratic Assignment Problem
- [QAP Wikipedia](https://en.wikipedia.org/wiki/Quadratic_assignment_problem)
- 8개의 department와 8개의 location이 있음
- objective = minimize flow costs between the placed departments
- flow cost is flow * distance
- optimal flow costs is 107(or 214)
- <img src="https://www.dropbox.com/s/2m64zmd3e114r2e/Screenshot%202019-03-29%2020.59.07.png?raw=1">

### 코드
```
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

Dist = pd.DataFrame(
    [[0, 1, 2, 3, 1, 2, 3, 4], [1, 0, 1, 2, 2, 1, 2, 3], [2, 1, 0, 1, 3, 2, 1, 2], [3, 2, 1, 0, 4, 3, 2, 1],
     [1, 2, 3, 4, 0, 1, 2, 3], [2, 1, 2, 3, 1, 0, 1, 2], [3, 2, 1, 2, 2, 1, 0, 1], [4, 3, 2, 1, 4, 2, 1, 0]],
    columns=["A", "B", "C", "D", "E", "F", "G", "H"], index=["A", "B", "C", "D", "E", "F", "G", "H"])

Flow = pd.DataFrame(
    [[0, 5, 2, 4, 1, 0, 0, 6], [5, 0, 3, 0, 2, 2, 2, 0], [2, 3, 0, 0, 0, 0, 0, 5], [4, 0, 0, 0, 5, 2, 2, 10],
     [1, 2, 0, 5, 0, 10, 0, 0], [0, 2, 0, 2, 10, 0, 5, 1], [0, 2, 0, 2, 0, 5, 0, 10], [6, 0, 5, 10, 0, 1, 10, 0]],
    columns=["A", "B", "C", "D", "E", "F", "G", "H"], index=["A", "B", "C", "D", "E", "F", "G", "H"])

T0 = 1500
M = 250
N = 20
alpha = 0.9

X0 = ["B", "D", "A", "E", "C", "F", "G", "H"]

New_Dist_DF = Dist.reindex(columns=X0, index=X0)
New_Dist_Arr = np.array(New_Dist_DF)

# Make a dataframe of the cost of the initial solution

objfun1_start = pd.DataFrame(New_Dist_Arr * Flow)
objfun1_start_Arr = np.array(objfun1_start)

sum_start = sum(sum(objfun1_start_Arr))

print(sum_start)

Temp = []
Min_Cost = []

for i in range(M):
    for j in range(N):
        ran_1 = np.random.randint(0, len(X0))
        ran_2 = np.random.randint(0, len(X0))

        while ran_1 == ran_2:
            ran_2 = np.random.randint(0, len(X0))

        xt = []
        xf = []

        A1 = X0[ran_1]
        A2 = X0[ran_2]

        # Make a new list of the new set of departments

        w = 0
        for i in X0:
            if X0[w] == A1:
                xt = np.append(xt, A2)
            elif X0[w] == A2:
                xt = np.append(xt, A1)
            else:
                xt = np.append(xt, X0[w])
            w = w + 1

        # print(X0)
        # print(A1, A2)
        # print(xt)

        new_dis_df_init = Dist.reindex(columns=X0, index=X0)
        new_dis_init_arr = np.array(new_dis_df_init)

        new_dis_df_new = Dist.reindex(columns=xt, index=xt)
        new_dis_new_arr = np.array(new_dis_df_new)

        # Make a dataframe of the current solution
        objfun_init = pd.DataFrame(new_dis_init_arr * Flow)
        objfun_init_arr = np.array(objfun_init)

        # Make a dataframe of the new solution
        objfun_new = pd.DataFrame(new_dis_new_arr * Flow)
        objfun_new_arr = np.array(objfun_new)
        sum_init = sum(sum(objfun_init_arr))
        sum_new = sum(sum(objfun_new_arr))

        rand1 = np.random.rand()
        form = 1 / (np.exp(sum_new - sum_init) / T0)

        if sum_new <= sum_init:
            X0 = xt
        elif rand1 <= form:
            X0 = xt
        else:
            X0 = X0

        Temp.append(T0)
        Min_Cost.append(sum_init)

        T0 = alpha * T0

print()
print("Final Solution", X0)
print("Minimized Cost:", sum_init)

plt.plot(Temp, Min_Cost)
plt.title("Cost vs Temp", fontsize=20, fontweight='bold')
plt.xlabel("Temp", fontsize=18, fontweight='bold')
plt.ylabel("Cost", fontsize=18, fontweight='bold')
plt.xlim(1500, 0)

plt.xticks(np.arange(min(Temp), max(Temp), 100), fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
```


### Reference
- [Simulated Annealing, SA, 담금질 기법 - 위키피디아](https://ko.wikipedia.org/wiki/%EB%8B%B4%EA%B8%88%EC%A7%88_%EA%B8%B0%EB%B2%95)
- [Simulated  Annealing](http://www.aistudy.com/neural/simulated_annealing.htm)
- [동전 뒤집기와 Simulated Annealing](https://koosaga.com/3)
