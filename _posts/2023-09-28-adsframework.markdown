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

### 용어 설명  

- `ADS`: Alerting and detection strategy framework (탐지 전략 및 알림 프레임워크)  
- `IR Team`: incident response team (이슈 대응 팀)  
- `ATT-CK`: Adversarial tactics, techniques and common knowledge (이상에 대한 분류 체계)  

  

<br>

### 잘하는 IR Team이란?  

잘하는 IR Team은 좋은 ADS Framework를 설계하고, 실행한다.  

이를 통해 낮은 오탐을 보이고 실제 문제 상황에 강한 시그널을 보이는 서비스를 제공한다.  

못하는 IR Team은 `미숙한 알림`으로 잦은 오탐을 발생시키고, 이는 고객에게 높은 피로를 준다.  

  

<br>

### 미숙한 알람이란?  

미숙한 알림 서비스는 크게 3가지 특징을 보인다.  



1. 미흡한 문서화  

> 문서화가 미흡한 서비스들은 대표적으로 아래와 같은 특징을 보인다.  
>
> - 적절하지 않은 수신자 목록  
> - 오탐이 어째서 발생할 수 있는지에 대한 설명 부재  
> - 실제 문제가 발생했을 때 수신자가 어떻게 대응하면 될지에 대한 가이드  

  

2. 알림에 대한 검증 프로세스 부재  

> 검증되지 알림은 아래와 같은 문제를 파생한다.  
>
> - 너무 좁은 시야의 탐지로 미탐  
> - 너무 넓은 시야의 탐지로 오탐  
> - 잘못된 가설 설정, 데이터 사용, 기술 스택 선택  

  

3. Beta와 Live에 대한 구분이 없음  

> Beta와 Live의 의미있는 구분을 위해서는 아래의 기준들이 수립되어야 한다.  
>
> -  알림의 이상 수준에 대한 구분 (ex. low, mediaum, high)  
> - 정탐률에 대한 임계 수준  

  

<br>

### ADS Framework

**ADS Framework**란 전체 서비스에 대한 정책이나 개념적인 프로세스가 아니다.  

개별 개별의 알림에 대하여, 알림의 생성자가 이것을 서비스화 하기 이전에 작성해야 하는 **문서 덩어리**이다.  

그리고 각 알림의 로직에 대한 변경이 있을때에는 필수적으로 ADS Framework도 변경하도록 강제해야 한다.  

  

ADS Framework는 다음 10개의 항목으로 구성되어 있다.  

1. Goal  

> 어떤 이상 상황을 탐지하려는지에 대한 한 줄 요약  

  

2. Categorization  

> 탐지 전략에 대한 분류이다.  
>
> 팔란티어의 경우 [ATT-CK](https://attack.mitre.org/)에서 정의된 이상 분류 체계를 따른다고 한다.  
>
> ATT-CK는 개인의 어뷰징보다는 조금 더 원론적인 `보안` 전략에 대한 분류 체계에 적합해 보인다.  
>
> 도메인에 따라 적절한 분류 체계를 새로 정의하는 것이 필요할 것 같다.  

  

3. Strategy Abstract  

> 

  

4. Technical Context  

> ㅇ  

  

5. Blind spots and assumption  

> ㅇ 

  

6. False Positive  

>  ㅇ

  

7. Validation  

> ㅇ

  

8. Priority  

> ㅇ  

  

9. Response  

> ㅇ

  

10. Additional Resources  

> ㅇ

  

<br>

### Case Study: 





