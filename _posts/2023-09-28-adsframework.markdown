---
title: 지속 가능한 탐지 알림 서비스를 만드려면  
date: 2023-09-28 00:00:03 +0000
categories: [Tips for Working, Detection]
tags: [alert, detection]
seo:
  date_modified: 2023-09-28 00:00:03 +0000
---

[참조: Palantir Blog - Alerting and Detection Strategy Framework](https://blog.palantir.com/alerting-and-detection-strategy-framework-52dc33722df2)  

위 글은 내가 사랑하는 기업. [팔란티어](https://www.palantir.com/)의 블로그에 17년에 포스트 된 글이다.  

이슈 대응의 프로세스가 지속 가능하기 위해 지켜야 하는 Framework에 대한 개념글이다.   

그런데 당신들... 정말로 이렇게 일하고 있는거야..?   

  

<br>

### <b>용어 설명</b>  

- `ADS`: Alerting and detection strategy framework (탐지 전략 및 알림 프레임워크)  
- `IR Team`: incident response team (이슈 대응 팀)  
- `ATT-CK`: Adversarial tactics, techniques and common knowledge (이상에 대한 분류 체계)  

  

<br>

### <b>잘하는 IR Team이란?</b>  

잘하는 IR Team은 좋은 ADS Framework를 설계하고, 실행한다.  

이를 통해 낮은 오탐을 보이고 실제 문제 상황에 강한 시그널을 보이는 서비스를 제공한다.  

못하는 IR Team은 `미숙한 알림`으로 잦은 오탐을 발생시키고, 이는 고객에게 높은 피로를 준다.  

  

<br>

### <b>미숙한 알람이란?</b>  

미숙한 알림 서비스는 크게 3가지 특징을 보인다.  

**(1) 미흡한 문서화**  

> 문서화가 미흡한 서비스들은 대표적으로 아래와 같은 특징을 보인다.  
>
> - 적절하지 않은 수신자 목록  
> - 오탐이 어째서 발생할 수 있는지에 대한 설명 부재  
> - 실제 문제가 발생했을 때 수신자가 어떻게 대응하면 될지에 대한 가이드  

<br>

**(2) 알림에 대한 검증 프로세스 부재**  

> 검증되지 알림은 아래와 같은 문제를 파생한다.  
>
> - 너무 좁은 시야의 탐지로 미탐  
> - 너무 넓은 시야의 탐지로 오탐  
> - 잘못된 가설 설정, 데이터 사용, 기술 스택 선택  

<br>

**(3) Beta와 Live에 대한 구분이 없음**  

> Beta와 Live의 의미있는 구분을 위해서는 아래의 기준들이 수립되어야 한다.  
>
> -  알림의 이상 수준에 대한 구분 (ex. low, mediaum, high)  
> - 정탐률에 대한 임계 수준  

  

<br>

### ADS Framework

**ADS Framework**란 전체 서비스에 대한 정책이나 개념적인 프로세스가 아니다.  

개별 개별의 알림에 대하여, 알림의 생성자가 이것을 서비스화 하기 이전에 작성해야 하는 **문서 덩어리**이다.  

그리고 각 알림의 로직에 대한 변경이 있을때에는 필수적으로 ADS Framework도 변경하도록 강제해야 한다.  

ADS Framework는 10개의 항목으로 구성되어 있다.  

<br>

**(1) Goal**  

> 어떤 이상 상황을 탐지하려는지에 대한 한 줄 요약  

<br>

**(2) Categorization**  

> 탐지하는 이상에 대한 분류  
>
> 팔란티어의 경우 [ATT-CK](https://attack.mitre.org/)에서 정의된 이상 분류 체계를 따른다고 한다.  
>
> ATT-CK는 개인의 어뷰징보다는 조금 더 원론적인 `보안` 전략에 대한 분류 체계에 적합해 보인다.  
>
> 도메인에 따라 적절한 분류 체계를 새로 정의하는 것이 필요할 것 같다.  

<br>

**(3) Strategy Abstract**  

> 탐지가 어떤 단계로 진행되는지에 대한 설명  
>
> 무슨 데이터를 사용? 탐지 로직은? 알림 조건은? 오탐을 줄이기 위한 처리는? 등등  

<br>

**(4) Technical Context**  

> 알림의 수신자가 이슈를 대응하는데 필요한 자세한 정보들  
>
> 가장 추상적인 영역인 것 같다. 탐지하는 방법론에 대한 설명을 비롯하여  
>
> 탐지 대상의 도메인적인 정보들, 혹은 사용하는 데이터에 대한 설명까지도.  

<br>

**(5) Blind spots and assumption**  

> 탐지시에 하고 있는 가정이 무엇이고, 이에 따라 어떤 미탐 영역이 있을 수 있는지.   

<br>

**(6) False Positive**  

> 어떤 예외 조건들로 오탐이 발생될 수 있는지. 혹은 발생한 사례들.  
>
> 오탐이 발생하면 재발하지 않도록 SIEM에 등록하여 탐지 로직을 보완해야 한다.  
>
> ** SIEM: Security Information and Event Management tool  
>
> - 관제 솔루션에 대한 일반적 용어인듯!  

<br>

**(7) Validation**  

> 기대하는 이상 상황에 대한 구체적 시나리오.  
>
> 코드 내 unit test와 같이, 어떤 단계를 거쳐 알림까지 이어지는지에 대한 설명.  

<br>

**(8) Priority**  

> 알림이 의미하는 문제의 심각도에 대한 분류  

<br>

**(9) Response**  

> 수신자의 대응 가이드.  
>
> 이를 잘 작성하여 수신자가 인수인계시에도 원할하게 활용하도록 한다.  

<br>

**(10) Additional Resources**  

> 추가적인 자료들이 필요하다면!  

​    

<br>

### Case Study: [github - palantir / ads-example](https://github.com/palantir/alerting-detection-strategy-framework/blob/master/ADS-Examples/002-Modified-Boot-Configuration-Data.md)  

**(1) Goal**  

> Windows를 사용하는 장치의 부팅 구성 데이터(BCD)가  
>
> 비정상적이고 악의적일 가능성이 있는 방식으로 수정된 경우 탐지  

<br>

**(2) Categorization**  

> ATT-CK > Defense Evasion > Disabling Security Tools  

<br>

**(3) Strategy Abstract**  

> - 윈도우 이벤트 로그를 사용하여 모든 부팅에 대한 BCD 데이터를 수집한다.  
> - 잘 알려진 설정과 금번 BCD를 비교한다.  
> - 비교한 뒤 차이가 있으면 알림을 발송한다.  

<br>

**(4) Techinal Context**  

> BCD란 무엇인가 주저리주저리  
>
> BCD란 어느 경우에 수정될 수 있다 주저리주저리  
>
> BCD 데이터 안에서 중요한 항목들은 아래와 같다 주저리주저리  

<br>

**(5) Blind Spots and Assumptions**  

> 탐지 전략은 아래의 상황들을 가정하고 있다.  
>
> - 원도우 이벤트 로그의 정합성이 맞다.  
> - 설정을 비교하는 탐지 로직에 문제가 없다.  
>
>   
>
> 가정이 틀리면 아래와 같은 미탐이 있을 수 있다. 예를들면,  
>
> -  윈도우 이벤트 로그 없이 BCD가 수정될 경우 미탐  

<br>

**(6) False Positive**  

> 여러 경우들에 오탐이 발생할 수 있다.  예를들면,  
>
> - 유저가 Window Insider Preview (WIP)에 등록했을 경우  
> - 유저가 정식으로 기능을 사용하여 설정을 변경했을 경우  

<br>

**(7) Priority**  

> - High: A,B,C,D 설정에 대한 변경이 있었을 경우  
>
> - Mediaum: E 설정에 대한 변경이 있었을 경우  

<br>

**(8) Validation**  

> 아래 코드를 실행해보시면 알림이 올거에요~!  
>
> `BCDEDIT /set nointegritychecks ON`  

<br>

**(9) Response**  

> 알림을 받으시면 아래 절차에 따라 대응을 부탁드립니다.  
>
> 1. 무슨 항목이 변경되었는지 확인  
> 2. E 항목만 변경되었을 경우 유저가 EIP에 등록한 오탐일 가능성이 높음  
>    1. 따라서 머시기머시기도 확인 요망  
>    2. 만약 어쩌구라면 보안 사건으로 전파 요망  
> 3. A,B 항목만 변경되었을 경우 어쩌구어쩌구...  
>
> ...

<br>

**(10) Additional Resources**  

> - [링크1]()
> - [링크2]()
> - [링크3]()

<br>

<br>













