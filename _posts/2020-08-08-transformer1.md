---
title: Attention is All You Need (NIPS 2017)
date: 2020-08-08 00:00:00 +0800
categories: [Paper Review, Language]
tags: [attention]
seo:
  date_modified: 2020-08-14 17:43:31 +0900
---

<br/>

블로그에 리뷰한 논문 중 Language Task를 다룬 적은 없었는데, 꼭 NLP가 아니더라도 다른 여러 분야의 핫한(?) 모델들의 기조가 되는듯 하여 자세히 들여다보고자 한다. 위 논문에 대한 리뷰는 3개의 포스트로 나누어 진행할 예정으로, (1) 우선 논문의 내용을 리뷰한 후, (2) PyTorch를 사용하여 모델을 구현하고, (3) 실제 게임내의 시퀀스 데이터를 사용하여 모델을 활용해보려고 한다.

<br/>

<img src="/assets/img/pr/transformer/transformer1.jpg">  

구글에서 2017년에 발표한 위 논문에 대한 가장 큰 직관은 제목에 있다. Attention이라는 하나의 메커니즘을 요리조리 사용하는 것이 모델의 전부이다. 특별히 복잡한 기교나 다양한 연산법이 들어가지도 않고, 오토인코더와 같이 단순한 형태의 구조를 취하여 Task를 해결한다. 모델에서 해결하는 과제는  English-to-German  /  English to-French Translation Task이다. 그러면 본격적으로 도대체 구글의 번역이 어떻게 이토록 `빠르고` `정확하게` 돌아가는지 확인해보자.  



# <b>Abstract</b>

기존에 어순이 다른 번역 과제를 해결하는 모델들은  복잡한 Recurrenct or Convolution 형태의 Layer들로 소스가 되는 언어를 Encoding하는 부분과 목적이 되는 언어를 Decoding하는 구조로 되어있었다. 가장 최신에 우수한 성능을 내는 모델은 이런 복잡한 Layer로 이루어진 인코더와 디코더를 어텐션으로 연결하는 형태를 띄고 있었다. 구글이 제안하는 모델 `Transoformer`는 단순히 연결의 용도로 Attention을 사용하는 것이 아니라, 인코더와 디코더의 모든 부분을 Attention Mechanism을 사용하였다. 이런 구조를 통해 얼마나 `빠르고` `정확하게` Transformer가 번역 과제를 수행하는지 뒤에서 자랑하겠다.

<br/>

#### <b> Introduction & Background</b>

- Rnn 구조의 모델은 병렬적인 학습이 불가능하기 때문에 학습에 큰 비용이 든다.
- Attention Mechanism은 Input의 길이에 대한 제약에서 비교적 자유롭다.
- Attention Mechanism은 시퀀스 내 단어간의 거리에 독립적으로 학습이 가능하다.

<br/>

# <b>Self - Attention</b>

[(좋은 설명 링크)](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)

논문에서 Attention Mechanism이 어떻게 돌아가는지 이해하기 위해서, 논문에서 사용된 `Self - Attention`의 정의를 우선 이해해야한다. 우선 단순하게 말해보자. n개의 인풋이 있다. 서로서로 누가 각자에게 중요한지 점수를 매긴다. 점수의 크기만큼 남의 데이터를 가중하여 새로운 나의 데이터를 만든다. 이렇게 가공된 n개의 output이 나온다. 컨셉적으로는 굉장히 간단한데 `점수를 매긴다`는 부분이 구체적으로 뭘까?

<br/>

**Transformer의 Self Attention Mechanism은 3가지 요소 (Key / Query / Value)로 이루어져 있다.**  

- 각 인풋마다 고유의 key / query / value를 가지고 있다.
- 나의 query를 남의 key와 inner product 함으로서 얻어진 값이 **나 -> 남 점수**가 된다.
- 나의 query로 얻어진 나 -> 남 점수들에 남의 value를 곱하여 **인풋 개수만큼의 벡터를** 얻는다.
- 나의 query로 얻어진 벡터들을 모두 element wise sum하여 얻어진 벡터가 **나의 output**이 된다.

<br/>

Mechanism을 이해하는데 헷갈리는 부분들을 미리 짚고 넘어가보자.  

- Key와 Query 의 길이는 항상 동일하다.
- Output과 Value의 길이는 동일하다. 
- 그러나 Key & Input & Ouput의 길이는 모두 달라도 상관없다.

<br/>

# <b>Model Architecture</b>

Transformer와 기존 RNN Based 모델의 차이점은 단순히 구조의 단순함이나 병렬적인 학습의 가능함에 있지 않다. 성능에 영향을 미치는 가장 결정적인 차이는 Encoding된 Input Data가 어떻게 Ouput 예측에 사용되는지에 있다.  

<img src="/assets/img/pr/transformer/transformer2.jpg"> 

**기존 RNN 모델의 경우**  

**Input** (X1,X2,X3,X4 .. Xn) 데이터가 고정된 벡터 C로 **Encoding** 된 후, 첫 **Output** Y1을 예측한다.  

그 후, Y2 예측에는 C와 Y1이,  Y3의 예측에는 C와 Y1,Y2가 사용된다.  

즉 Input이 하나의 고정된 벡터로 인코딩 된 후, 예측을 거듭할 수록 영향력을 잃게 된다.  

<br/>

**Attention이 사용된 Transformer의 경우**  

각 Output Y(i)에 대해서 Attention이 적용된 서로 다른 Input Encoding C(i)를 얻게 된다.

<br/>

## <b>Encoder and Decoder Stacks</b>

> **Encoder:** The encoder is composed of a stack of N = 6 identical layers. Each layer has two
> sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. 

<br/>

- **Positional Encoding**
  - input sequence의 단어 순서를 고려하기 위해서 추가된 Layer
  - non-trainable vector로서 position마다 정해진 형태의 값을 더한다
  - [What is the positional encoding in the transformer model?](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model) 

- **Self-Attention Layer**
  - Input의 어떤 단어에 더 초점을 맞추어 Encdoing을 진행할 지 결정하게 되는 Layer
  - Multi-head로 본다는 것은 Convolution을 multi-channel로 하는 것과 같은 개념이다.


- **Pointwise Feed Forward Layer**

  - Transformer에서는 residual connection을 적용한 두 층의 linear layer 사용

  - > We employ a **residual connection** around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

<br/>

<img src="/assets/img/pr/transformer/transformer3.jpg"> 

<br/>

> **Decoder:** The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
> sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
> attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
> around each of the sub-layers, followed by **layer normalization**. 

<br/>

- **Layer normalization**

  - layer normalization과 일반적으로 사용되는 batch normalization은 구별해야한다.

    <img src="/assets/img/pr/transformer/transformer4.jpg"> 

  - 식과 같이 batch normalization이 여러 example의 동일한 feature에 대한 정규화라고 한다면,

    layer normalization은 하나의 example 내에서 여러 feature들을 정규화하는 것이다.

<br/>

> We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This **masking**, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

<br/>

- **Masking**
  - Encoding의 경우, 모든 input을 동시에 집어넣어서 병렬적으로 수행되지만, Decoding을 그런 방법으로 수행했을 때에는 아직 문장에서 등장하지 않은 미래의 단어를 참고하게 된다. 따라서 Self-Attention을 수행할 때, 이미 번역을 수행한 Output들에 대해서만 Score를 계산한다.  

<br/>

## <b>Attention</b>

<img src="/assets/img/pr/transformer/transformer5.jpg"> 

**Scaled Dot-Product Attention**

key / query / value를 통해서 self-attention을 수행하는 방법이다. 

<img src="/assets/img/pr/transformer/transformer6.jpg"> 

1. key 길이에 대한 제약을 주고자 dot - product의 값을 벡터 길이에 대응하는 값으로 나눠준다. 
2. 그 이후 input 길이만큼의 (value*query) 값들을 softmax를 통해 0과1 사이의 수로 바꿔준다.
3. 마지막으로 고유의 value 값들에 softmax 결과물을 곱해줌으로서 Self-Attention이 끝난다.

<br/>

**Multi-head Attention**

문장의 word간의 다양한 관계를 capture하기 위해 위의 Self-Attention을 여러 겹으로 동시에 수행한다.  

- 논문에서는 8개의 multi-head를 사용했다. 따라서 8쌍의 (K/Q/V)에 대한 initialization이 필요하다.
- Feed-Forward 이후에 8개의 output vector를 concat한 후, Linear Layer에 태워 원래 벡터 크기로 줄여준다. 

<br/>

구글에서 2017년에 발표한 논문 `Attention is all you need`에서 고안된 모델 `Transformer`에 대하여 살펴보았다. 논문에는 물론 Transformer가 기존 모델에 대비하여 복잡도나 성능의 측면에서 얼마나 우수한지에 대하여 이론적이고 실험적으로 충분히 설명되어있다. 해당 부분들에 대하여 자세히 리뷰하는 것도 물론 중요하겠지만 그보다는 직접적으로 `Pytorch Implementation`을 우선적으로 수행해보고자 한다.

<br/>

#### <b>자료</b>

- [논문](https://arxiv.org/pdf/1706.03762.pdf)
- [코드](https://github.com/tensorflow/tensor2tensor#language-modeling)
  
- 이미지
  - [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/) 

