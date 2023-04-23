---
title: 제품 분석 / Palantir Foundry
date: 2023-03-25 00:00:03 +0000
categories: [Chat, Review]
tags: [review]
seo:
  date_modified: 2023-03-25 00:00:03 +0000
---

<br>

<img src="/assets/img/chat/palantirfoundry/palantirfoundry1.jpg">

파운드리라는 단어는 반도체 공장에나 쓰는 말 아니었나..?  

Ontology..? Topology 친구인가..? 수상한 미국 회사 <b>Palantir</b>의 제품. Foundry를 분석해본다.  

팔란티어(Palantir)는 실리콘밸리에 있는 2003년에 설립된 꽤나 성숙한 SW 회사이다. ZERO TO ONE의 저자인 피터 틸이 페이팔의 이상 탐지 노하우를 데이터 분석 제품화하여, 다시는 9.11 같은 참사를 발생시키지 않겠다며 설립했다고 한다. 팔란티어의 홈페이지 메인에는 `고객님의 서비스 운영 과정의 모든 의사결정을 AI에 기반하여 내릴 수 있도록 지원`하겠다는 미션이 쓰여져있다.  

요즘 핫한 Chat-GPT 같은 친구들은 대부분 다양한 Task를 유연하게 학습하고, 평균의 정답에 수렴하는 General AI이다. 그에 반해 팔란티어가 제공하는 AI는 구체적이고 독특한 문제들을 해결하는 Narrow AI의 성격을 띈다. Foundry는 각 기업이 가지고 있는 다양한 데이터를 (가치 있는 형태로) 연결하고, 그 안에서 각 기업이 내려야 하는 (복잡한) 의사결정을 지원한다.  

<br>

> 문제를 정의하고, 필요한 데이터를 모으고, 분석을 통해 통찰을 얻는다.  
>
> 모델링 결과에 입각하여 데이터 기반의 의사결정을 내린다.  

이상적인 Data-Driven 조직의 모습이라 볼 수 있다. 요새는 대부분의 회사가 이런 형태의 조직을 추구하지만, 실제로 이렇게 일하는 것은 쉽지 않다. 단순히 어떤 정책이나 절차가 있다고 데이터 기반의 의사결정이 갑자기 이뤄지지 않는다. 데이터란 불완전한 인간이란 존재가 가공한 불완전한 정보이기 때문이다. 꼭 어떤 비정형의 행동 정보만이 인간의 데이터가 아니다. 기계가 쌓는 데이터라고 해도 그 데이터의 스키마를 정의하고, 저장되는 테이블에 이름을 붙여준 것은 인간이다. 따라서 데이터의 생산이나 사용의 주체가 아니면서 그 데이터를 제대로 이해하고 연결하는 것은 (아주) 아주 어려운 일이다. Palantir Foundry를 한 줄로 정의하면, 개 쩌는 백오피스이다. 백오피스의 미션, 기능, 역할을 이해하기 위해 Foundry를 대표하는 Ontology 철학을 이해해보자.  

<br>

#### <b>The Foundry Ontology</b>  

Ontology의 사전적 의미는 `사물과 사물 간의 관계 및 여러 개념을 컴퓨터가 처리할 수 있는 형태로 표현하는 것`이다. 팔란티어는 데이터 기반의 의사결정에 필요한 모든 요소들을 3개 Layer로 표현(Ontology)하며, 그 안에서 Closed-Loop Operation을 구현한다고 한다. Closed Loop Operation이란 사람의 개입 없이 컴퓨터가 피드백을 취합하여 다음 작업에 그것을 반영하는 것을 의미한다. 팔란티어가 추가하는 Closed Loop는 (데이터 -> 모델 -> 의사결정 -> 결과 -> 데이터)의 구조이다. 단순히 데이터 기반의 의사결정을 지원하는 것이 아니다. `데이터 기반의 의사결정을 자동화` 하겠다는 것이다. 이 얼마나 가슴이 웅장해지는 미션인가...  

그렇다면 Closed Loop Operation을 구성하는 3개 Layer를 각각 조금 자세히 살펴보자.  

<br>

#### <b>Ontology Layer 01 / Semantic</b>  

Semantic Layer는 비즈니스 로직을 프로그래밍적으로 정의하는 기능이다.  

Semantic Layer는 기본적으로 객체 지향적 프로그래밍의 컨셉을 취한다. 비즈니스를 구성하는 요소들을 객체(Object)로 정의하고, 각 객체에 속성(Properties)를 설정하고, 그 관계(Link)를 표현한다. 예를들어 승객(Passenger)과 항공편(Flight)이라는 객체를 만들어보자. 승객에게는 다양한 속성이 있을 수 있다. 이름, 성별, 연령 등등. 항공편에게도 다양한 속성이 있을 수 있다. 항공사, 출발 시간, 출발 장소 등등.   사실 여기까지는 단순히 테이블을 생성하고 스키마를 정의했구나 정도로 생각이 드는데... 진정한 Ontology의 묘미는 객체간의 Link를 어떻게 관리하는가에서 나오겠구나 싶었다. Ontology의 Link란 단순히 테이블간의 연결 관계 (primary key <-> foreign key)를 나타내는 도구가 아니다. 그보다는 여러 비즈니스 요구사항을 반영한 데이터 마트를 관리하기 위한 도구로 느껴졌다.  

<br>

 <img src="/assets/img/chat/palantirfoundry/palantirfoundry2.jpg">

<center><small>예를들어 이렇게 표현해 볼 수 있지 않을까..?</small></center>

- Passenger와 Flight 두 종류의 객체(Object)가 존재한다.  
- 각 객체를 구분하는 여러 속성(Properties)가 존재한다.  
- 각 Passenger가 어떤 Flight에 포함됬는지에 대한 `Passenger -> Flight` 관계(Link)가 존재한다.  
- Passegenr 객체를 상속한 VIP라는 객체가 존재한다.  

<br>

#### Ontology LAYER 02 / Kinetic  

Kinetic Layer는 비즈니스 로직을 분석하는 개선하는 기능이다.  

Kinetic Layer는 객체들을 사용하여 실제로 action을 수행하는 단계이다. Foundy Ontology의 object, property, link를 어떤 형태로든 변화시키는, 유저가 정의한 모든 로직을 Action이라고 정의한다. 무언가 변화가 있어야 액션이기 때문에 CRUD에서 R을 제외한 CUD가 Action이라고 볼 수 있다. 

<img src="/assets/img/chat/palantirfoundry/palantirfoundry3.jpg">

Foundry에서는 누구나 쉽게 CUD API를 제작할 수 있도록 지원한다. API의 이름을 정하고, 액션의 타입을 정하고, 실제로 그것이 작동할 Rule을 모두 기능을 통해 구현할 수 있다. Foundry의 고객들은 API나 DB 같은, 겁나는 컴퓨터 용어들을 배우지 않고도 사용성을 극대화하기 위해 굉장한 편의를 제공한다. 각 회사가 가지고 있는 고유한 Business Process를 Foundry의 개념과 기능으로 정의하고, 이것에 익숙해진 고객들이 어느날 갑자기 다른 소프트웨어로 이를 대체한다는 것이 가능할까..?  

또한 Kinetic Layer의 기능으로써 Process Mining이라는 개념이 등장한다. 이것도 Palantir에서 정의한 것인가 했더니 실제로 있는 학문(?)의 한 분야이다. IT System에서 쌓이는 로그성 데이터를 분석하여 비효율을 발견하고 Business Process를 재 설계하는 단초가 되는 업무라고 한다. 재밌을 거 같다.  

<br>

#### <b>Ontology LAYER 03 / Dynamic</b>  

Dynamic Layer는 비즈니스 로직을 시뮬레이션하고 인과 관계를 이해하기 위한 기능이다.  

사실 여기는 조금 추상적인 설명 위주라 잘 이해가 안된다.  

<br/>





