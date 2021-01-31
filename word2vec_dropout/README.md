# World Embedding and Dropout

## Method

Original skip-gram word embedding:

![](./figures/skip_gram_0.png)

word embedding with dropout:

![](./figures/skip_gram.png)

## Result

If we embed the simple network:

![](./figures/WechatIMG249.png)

Using the traditional method, we have:

![](./figures/WechatIMG253.png)

And the embedding vector is:

![](./figures/WechatIMG252.png)

But, if we use dropout, we have a much better result:

![](./figures/WechatIMG251.png)

And the vectors are:

![](./figures/WechatIMG250.png)

Two groups are separated in both code and space.