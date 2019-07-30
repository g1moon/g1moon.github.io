---
layout: post
title: "4-CNN-2.CaseStudy"
subtitle: "4-CNN-2.CaseStudy"
categories: data
tags: deeplearningai
comments: true
---
<img src="https://i.imgur.com/RXU39la.png">
# 2.케이스 스터디

# 케이스 스터디

## 왜 합성곱을 사용할까요?

(C4W1L11 Why Convolutions)

- cnn은 신경망의 변수가 작게 나타나는데 그 이유는 아래와 같고 → 훈련셋이 작아지고 오버피팅 방
    - 파라미터를 서로 공유할 수 있음→ 그러니까 왼쪽에서 유의미하게 사용된 속성 검출기는 오른쪽 아래에서도 유용하게 사용 할 수 있음( 값이 달라 조금 다르게 보일 수 있 @지만 (10과0) 두 곳에서 모두  좋은 효과를 줌
    - 희소 연결성(sparsity of connection) 오른쪽 아웃풋의 첫번쨰 초록 동그라미 값 0을 보면 이것은 첫번째 인풋 데이터의 3x3(필터사이즈)인 곳에서만 연결되어있어 
    즉... 36(4x4)의 출력 값은 입력중 9(3x3필터사이즈)개만 연결되어있음.. 나머지 픽셀 값들은 결과값에 영향을 주지 않음

- conv layer 와 fc 레이어는 w라는 변수를 갖게되고 b편향을 갖는다 (무작위로 w와b를 초기화해서 cost J를 계싼할 수 있음 m으로 나눠 → 평균) → 그라디언트디센트, 모멘텀 경사하강, RMsprop를 사용해 최적화
- 또 완전 연결 층 뒤에는 소프트맥스 출력값인 y_hat이다.

## computer vision - case study

- 

### LeNet - 5 (예전 알고리즘)

- avg poolingd을 많이 사용했음
- 높이와 너비는 낮아지고 채널은 늘어남
- conv/ pool/ conv /pool /fc /fc/ output
- 시그모이드 층을 풀링층 뒤에 위치시켰음 (activation fuction)

### AlexNet

- Lenet과 매우 유사하지만 더 큼
- 더 많은 히든 레이어와 더 많은 데이터로 훈련하기떄문에 더 좋은 성능이 나옴
- 리넷과 구별되는 것은 relu를 사용한다는 것

### vgg16 (16개의 가중치를 갖는)

요즘 기준으로도 큰편 그리고 장점은 꽤나 균일하다는 것 
풀링층이 높이와 너비를 줄여주고, 획일성이 주는 단점은 훈련시킬 변수가 많다 

- 교수가 좋아하는 부분은 깊이가 깊어질 수록 보여지는 패턴이 // 풀링층에서는 높이와 너비가 매번 절반으로가서 // 합성곱층에서는 채널수가 매번 두배가량// 수치가 커지고 작아지는 것이 상당이 체계적으로 나타나져 있고 → 이러한 점이 매우 매력적임
- cov = 3x3 필터이고, stride = 1 이고 same(same conv)으로 간다(모든 필터는 3x3)
- 그리고 max-pool은 2x2 이고 , stride = 2이다
- 64개의 필터를 가진 두개의 conv layer를 사용한다는 의미이다
- 그다음 풀링레이어에서 사이즈가 줄어들고
- 128개를 갖는 두개의 컨브레이어 → 128→ 256 .... 7x7x512 → fc(4096) → fc(4096) → softmax

## Resnet

(C4W2L03 Resnets)

- 아주 깊은 신경망을 학습하지 못하는 이유는 배니싱 그라디언트 or 폭발적 증가 떄문
- 하지만 레스넷에서는 skip connection 으로 이 문제를 해결
- 모든 층을 지나는 연산 과정을 main path
- 

    ![](https://imgur.com/GDVjwFj)

![](https://imgur.com/1A0CW3Q)

- 잔여 블럭이라는 것으로 구성
- a^[l]의 정보가 short cut을 따라서 신경망의 더 깊은 곳으로 갈 수 있음  →
- 이것으로 마지막 식이 사라지고 a[l+2] = g() 로 되고
- 두번쨰 층으로 가는 이유는 relu 전에 더해지기 떄문
- 리니어 전에 나가고 // 릴루 전에 들어감(리니어 연산 뒤에 삽입되고 릴루 전에 들어감)

- 5개의 잔여 블럭이 합쳐진 것이고 이것이 resnet
- 하지만 레스넷에서는 신경망이 깊어져도 계쏙 에러가 줄어드는 든다..
- 활성값 x 또는 중간 활성값을 취하는 것으로 → 그라디언트 베니싱 문제를 해결해주고 → 더 깊은 신경망을 성능 저하 없이 실행시켜줌

    main path image 

    ![](https://imgur.com/Tk8y1ue)

    ![](https://imgur.com/mvlr4q3)

비선형 적용 전에 → l+2 에 l 것을 더해주고 // 비선형성을 적용시켜줌

ReLUs [15] are applied to the output of the n × n conv layer.

- skip connection / short cut
    - a[l]을 더해서 다시 활성화 함수로 넣는 부분까지를 residual block이라고 한다
    - shortcut 혹은 skip connection 은 a[l]의 정보를 더 깊은 층으로 보내기 위해 일부 층을 검는 역할

        ![](https://imgur.com/BOIiJ3N)

    ![](https://imgur.com/1SEKJtY)

    - 또한, 경험적으로 층의 개수를 늘릴 수록 훈련 오류는 감소하다가 다시 증가합니다. 하지만 이론 상으로는 신경망이 깊어질 수록 훈련 세트에서 오류는 계속 낮아져야합니다. 하지만 ResNet 에서는 훈련오류가 계속 감소하는 성능을 가질 수 있습니다.

    ## 왜 ResNets 이 잘 작동할까요?

    ![](https://imgur.com/1SEKJtY)

    ![](https://imgur.com/MSZg9Yr)

    - 만약 l2규제나 가중치 붕괴를 사용하면 w^[l+2]의 값이 감소 한다.
    - 만약 가중치 붕괴를 b도 적용하면 마찬가지
    - 만약 w,b가 0이라면 초록x 부분 사라지고 그냥 a[l]
    - 그래서 항등 함수는 잔여 블록의 훈련을 용이하게 함
    - 그리고 a[l+2] 와 a[l]이 같게 되는 이유는 skip connection 떄문이고

    ![](https://imgur.com/QpO5PLd)

    - same conv가 이것을 가능하게해줌
    - 만약 다르면 새로운 w_s를 추가하는데 —> 여기서 w_s는 256x128의 행렬이라
    - w_s와 a[l]의 곱은 256차원이 된다

    ![](https://imgur.com/JERZGLM)

    - 기존의 plane 에서 resnet으로 하려면 skip connection이 필요하고, 3x3의 필터들로 구서오디어 있고, same conV을 해주기 떄문에  차원이 유지되고 → 그래서 z[l+2] + a[l] 이 성립이 되는 것이다
    - 가끔 풀링레이어나 유사한게 있으면 차원을 조정해야하는데 → 이전 슬라이드에서 W_s같은것을 말한다
    - conv conv conv pooling / conv conv conv pooling / conv conv conv pooling 이 매우 흔하고 끝에서는 fc layer와 softmax를 이용한 예측값이 나옴

    ## why does a 1by1 convolution do?

    ![](https://imgur.com/pMrJgcd)

    - 1바이1 conv가 하는 일은 36개의 위치를 각각 살펴보고 (6x6)
    - 그 안의 32개의 숫자를 필터의 32개 숫자와 곱해
    - 그리고 relu하고
    - 36개중 하나에 노란 긴 조각이 생기고
    - 거기에 32개의 숫자를 이 한조각의 숫자와 곱해주면 → 하나의 숫자만 남는다 그리고
    - 이렇게 초록 점으로 남게 된다.

     

    - 하나의 필터가아니라 여러개의 필터라면 여러개의 유닛으로 입력 받아 한 묶음으로 되고
    - 출력은 6 6 의 필터 수만큼 갖게 된다
    - 32개의 숫자를 입력으로 받고 → 필터 수 만큼 출력
    - 1x1 conv 를 network in network 라는 아이디어로 표현된다.

    ![](https://imgur.com/5MVEhG9)

    - 만약 높이와 넓이를 줄이려면 풀링을 하면 되지만
    - 채널을 줄이려면 어떻게 해야할 까??? (이전의 경우 대부분 채널수가 증가했음)
    - 방법은 1곱1 필터를 32개 쓰면 채널수를 줄일 수 있음
    - 1곱1 합성곱의 효과는 네트워크에 비선형성을 더해주고, 하나의 층을 더해줘 , 더 복잡한 함수를 학습할 수 있음 (채널이 유지되니까) → 28 28 192 를 입력받아 —> 28 28 192를 출력할 수 있음
    - 1곱1 합성곱은 채널을 조절할 수 있고, 네트워크에 비선형성을 더해줘 도움을 준다

        ![](https://imgur.com/LtaXlhZ)

        # C4W2L06 Inception Network Motivation

        Inception 네트워크의 아이디어

        인셉션 모델은 전부 써라 , 필터의 사이즈, 풀링의 사이즈를 고려하고 싶지 않다면 모두 써서 모델이 스스로 학습하게 할 수 있게

        ![](https://imgur.com/pZbQGbA)

        ![](https://imgur.com/DUW3IcE)

        - 너무 비싼 코스트 120m... 하지만 1곱1로 줄일 수 있음 한번 봐보자 ~
        - 

        - bottle neck

            1곱1 필터로 채널수를 조정 (1 1 192)를 

            ![](https://imgur.com/V83zgAr)

            ## C4W2L07 Inception Network

            (googlenet)

            - channel concat 은  이전 강의에서 봤던 위 그림과 같음
            - 그리고 이 체널컨켓이 끝나면 1 inception

                ![](https://imgur.com/V1LdjZ3)

                - side branch 는 hidden layer을 가지고 prediction을 하는 것
                - 그래서 side branch에 softmax 가 있음( 원래 인셉션 넷의 맨 뒤에 fc layer랑 softmax가 있음)
                - 은닉층이나 중간층에서 예측을 해도 이미지의 결고를 예측하는데 그렇게 나쁘지 않음
                - 인셉션 네트워크에 정규화 효과를 주고,,, 네트워크의 과대적합을 방지해줌

                ![]()
