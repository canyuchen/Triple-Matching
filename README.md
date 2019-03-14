# Triple-Matching


## Problem Definition

Given a sentence that describes the financial situation of a company in the past few years.
> * e.g. sentence1: \
> 2013年度、2014年度和2015年1-6月，发行人应付账款余额分别为2306万元、635万元和70118万元。

We extract some entities , include time, attribute, and value.
* **times**:  2013年度、2014年度、2015年1-6月
* **attributes**: 应收账款   (attributes might be not complete)
* **values**: 2306万元、635万元、70118万元

We define a triple is consisted of three ordered components: **[time, attribute, value]**. 
If the sentence states that at time t, the value of attribute a was v, we say triple **[t, a, v]** is a correct triple.
e.g.   all correct  triples in sentence1:
>【2013年度、应付账款、2306万元】\
>【2014年度、应付账款、635万元】\
>【2015年1-6月、应付账款、70118万元】

**The problem is: given a sentence and entities in that sentence, 
extract all correct triples.**












