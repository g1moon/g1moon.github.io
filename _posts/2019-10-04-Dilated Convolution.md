---
layout: post
title: "D. Neural Language Models"
subtitle: "Neural Language Models"
categories: data
tags: dl
comments: true
---
# Dilated Convolution 

![](https://user-images.githubusercontent.com/44131043/66139163-e53d4400-e63a-11e9-9864-f4e5840f3fe3.png)

- 중간중간 간격이 있는 모습을 확인 할 수 있음

![](https://user-images.githubusercontent.com/44131043/66139164-e53d4400-e63a-11e9-89a0-66287c68198a.png)

- contextual information 이 중요한데 (애매한 그림 같은 경우 주변의 환경이나 배경이나 동작 자세 정보를 알면 정확한 판단이 가능한 경우가 많음)
- object detection , segmentation 에서는 더 중요
- 하나의 필터가 contextual information 을 어덯게 받아드리려면 → 주변의 픽셀들까지 전부 입력으로 받아드려 → 반응치를 계산할 수 있어야함 
→ 그러기 위해서는 receptive filed 를 확장해야함(조금 더 넓은 영영의 정보를 고려했다)

![](https://user-images.githubusercontent.com/44131043/66139170-e5d5da80-e63a-11e9-899b-456c3b47ea32.png)

- receptive filed를 확장하기위해
1. kernel size를 늘리는 경우가 있고
2. 계속 층을 적층시킴
- 실제로 이렇게 하면 작은 커널을 가진 걸 계속 적층하면 모서리 부분이 계싼이 잘 되지 않음
- 되긴 하는데 생각하는 것 처럼 꼼꼼히 되지 않음

![](https://user-images.githubusercontent.com/44131043/66139168-e53d4400-e63a-11e9-8e9d-57125fa4a795.png)

그래서 등장한 dilated cnn

---

---

- 컨볼루션 사이에 다운 샘플링 할때 업샘플링 하는 과정에서 컨티뉴어스하지 않은 리스폰스 맵이 나오는데
- 근데 dilated cnn으로 서브 샘플링 , 업샘플링 없을때는 조금 더 자연스러운 결과가 나온다는 결과

![]()

---
