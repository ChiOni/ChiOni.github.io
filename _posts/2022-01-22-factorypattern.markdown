---
title: Factory Method Pattern (python)
date: 2022-01-22 00:00:03 +0000
categories: [Python]
tags: [design pattern]
seo:
  date_modified: 2022-01-22 00:00:03 +0000
---

<br/>

디자인 패턴 (Design Pattern)은 소프트웨어를 개발하는 과정에서 자주 발생하는 문제에 대한 해결책을, 꺼내어 쓸 수 있는 형태로 템플릿 화 해놓은 것을 의미한다. 발생할 수 있는 문제가 다양한만큼, 만들어 놓은 해결책도 다양하다.  

크게는 (생성 / 구조 / 행위) 세 분류로 디자인 패턴을 나눌 수 있고, 각 분류안에서도 다양한 해결책과 템플릿이 존재한다. 이것들을 다 알고자 목표하는 것은 아니고, 지금 마주한 문제를 풀기에 적절한 해결책으로 보여지는 `팩토리 메소드 (Factory Method) 패턴`에 대하여 이해하고 적용해보려 한다.  

생성 디자인 패턴은 말그대로 객체(Class)를 어떻게 생성할 것인가에 대한 해결책이다. 팩토리 메소드 패턴은 생성 디자인 패턴의 파생 상품 중 하나인데, `객체를 만들어내는 부분을 서브 클래스 (Factory)에 위임하는 것`이 가장 큰 특징이다.  

직관을 주는 여러 글을 읽어보았을 때, 팩토리 메소드 패턴은 무엇이든 만들어 낼 수 있는 재료가 있으나 아직 구체적으로 무슨 물건을 만들지 결정하지 않았을 때 적절한 패턴이라고 한다. 팩토리 메소드 패턴의 목적은 새로운 물건을 만들고자 결정했을 때, 기존의 물건을 만들어내는 메인 프로그램의 변경을 최소화하는 것이다.  

<br/>

만약 우리가 어떤 의미를 갖는 다양한 리스트 형태의 데이터를 가지고 있다고 생각해보자.  

그리고 이것들을 다양한 알고리즘을 통과시켜 다양한 형태로 요약하고 싶은 니즈가 있다고 생각해보자.  

```python
class ListData:
    def __init__(self, name, value):
        self.name  = name
        self.value = value
        
class DataSummarizer:
    def get_summary(self, data, algorithm):
        if algorithm == 'mean':
            import statistics
            summary = statistics.mean(data)
            return summary

        elif algorithm == 'median':
            import statistics
            summary = statistics.median(data)
            return summary

        else:
            raise ValueError(algorithm)
        
list_data  = ListData(name = 'test1', value = [1,2,3,4,5])
summarizer = DataSummarizer()

print(summarizer(list_data, 'mean'))
print(summarizer(list_data, 'median'))
print(summarizer(list_data, 'MAD'))
```

위 코드를 돌려보면 우리가 미처 정의하지 못한 MAD 알고리즘을 수행하려고 할 때 에러가 발생할 것이다. 만약 우리가 관리하는 알고리즘이 100개가 넘는다면? 어느날 데이터가 nested list 형태로 들어오게 된다면? 확실히 위의 코드 구조를 계속 가져가는 것은 괴로운 일이 될 것 같다.  

<br/>팩토리 메소드 패턴의 시작은 우선 저 알고리즘들을 분리하는 것에서 시작한다.  

```python
class MeanSummarizer:
    def get_summary(self, data):
        import statistics
        summary = statistics.mean(data)
        return summary
    
class MedianSummarizer:
    def get_summary(self, data):
        import statistics
        summary = statistics.median(data)
        return summary
```

<br/>

그리고 여기서 다양한 알고리즘들을 관리하는 것을 목적으로 하는 팩토리라는 개념이 등장한다.  

```python
class SummarizerFactory:

    def __init__(self):
        self._algorithms = {}

    def register_name(self, name, algorithm):
        self._algorithms[name] = algorithm

    def get_summarizer(self, name):
        algorithm = self._algorithms.get(name)
        if not creator:
            raise ValueError(format)
        return creator()


factory = SummarizerFactory()
factory.register_name('mean', MeanSummarizer)
factory.register_name('median', MedianSummarizer)
```

<br/>

맨 처음 살펴본 코드에서 DataSummarizer는 알고리즘을 관리하는 역할도 수행했었지만, 이제는 그 역할을 SummarizerFactory에게 위임하기로 한다. 이제 DataSummarizer는 어떤 알고리즘이 존재하는지 신경쓰지 않고 단순히 요청받은 name에 대한 summary를 수행하는 존재가 되었다.  

```python
class DataSummarizer:
    def summarize(self, data, name):
        summarizer = factory.get_summarizer(name)
        return summarizer.get_summary(data)
```

<br/>

위와 같은 구조의 코드 관리는 여러 작업자가 각개로 알고리즘을 개발하고 있을 때 유용할 것으로 보인다.  어차피 내가 무언가 작업을 하고 있더라도 `factory.register_name`을 통해 관리자에게 컨펌을 받기 전에는 메인 프로그램에 영향을 미치지 않기 때문이다.

<br/>

만약 어느날부터  ListData 객체의 데이터가 이상한 형태로 들어온다고 하더라도, 개별 알고리즘을 수정할 필요가 없다. 우리가 해줘야 하는 작업은 다양한 형태의 데이터를 알고리즘에 들어가기 적합한 형태로 변환해주는  `convert_to_list` 함수(혹은 객체)를 만들어서 한 줄 추가해주기만 하면 된다.  

```python
class DataSummarizer:
    def summarize(self, data, name):
        summarizer = factory.get_summarizer(name)
        list_data  = convert_to_list(data)
        return summarizer.get_summary(list_data)
```

<br/>

복잡하고, 앞으로도 더 복잡해질 수 있는 여지가 있는 프로그램을 관리하기에 적합한 팩토리 메소드 패턴에 대하여 살펴보았다. 사실 디자인 패턴이라는 것이 fancy해 보이기는 했지만, 실제 우리가 겪고 있는 문제에 그대로 대입하기에는 또 애매한 부분들이 있었다. 그렇지만 클래스간의 결합도를 낮추기 위해 팩토리 메소드가 취하고 있는 개념은, 복잡해지는 코드를 if-else로 땜빵하지 않기 위해 늘 유념할 필요가 있을 것 같다.  

<br/>

<b>참조</b>

- [The Factory Method Pattern and Its Implementation in Python](https://realpython.com/factory-method-python/)
- [Factory Pattern. When to use factory methods?](https://stackoverflow.com/questions/69849/factory-pattern-when-to-use-factory-methods#:~:text=It%20is%20good%20idea%20to,were%20specified%20by%20sub%2Dclasses)

<br/>

