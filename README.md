### Topics for Assignment3
* Loss Function: Triplet-loss
* Other Arch Decisions: ResNet
* Regularization: DropOut

## Loss Function: Triplet-loss
![triplet loss](https://www.researchgate.net/publication/303794993/figure/fig14/AS:372579471773698@1465841276337/An-illustration-of-training-the-retrieval-CNN-model-by-adding-a-triplet-loss-in-the.png)
>It's a technique of using triplet images to train a network for face recognition
>Triplet consists of 
>* Anchor image (A: ground truth)
>* Positive image (P: image of same person as anchor)
>* Negative image (N: image of different person from anchor)

#### Why use Triplet loss ?

##### Problem statement 
"CNN are poor for 1 image training but for face verification/recognition our training data is very limited"
* Face verification is an 1:1 problem where the network has to predict whether the input image, name/ID are valid
* Face recognition is 1:k problem where ther network has to predict whether the input image matches one from database

##### Solution
Use eucleadian distance between two images to judge degree of similarity between the two.
We can take output from deep/last layer and calculate norm of difference between the two images o/p
$$d(img1,img2) = {||f(x^{1}-f(x^{2})||_2}^2$$ 
hence for triplet loss
$${||f(A) - f(P)||}^2 + \beta \le {||f(A) - f(P)||}^2 $$
Minimize the loss to ZERO
$$L(A,P,N) = max({||f(A) - f(P)||}^2 + \beta - {||f(A) - f(P)||}^2,0 )$$
$$J = \sum_{i=1}^{i=m}L(A^{(i)},P^{(i)},N^{(i)})$$
_____

## Other Arch Decisions: ResNet

#### What is ResNet ?
![resnet](https://i.imgur.com/HwKKi8V.png)
>ResNet or Residual Network is an outcome of "skip connection" or "short-cut" input from shallow layers in the network directly to deeper layers.

$$z^{(l)} =w^{[l+1]}a^{[l]}+b^{[l+1]}$$
$$a^{[l+1]}=g(z^{[l+1]})$$
$$z^{(l+2)} =w^{[l+2]}a^{[l+1]}+b^{[l+2]}$$
_without skip connection_ 
$$a^{[l+2]}=g(z^{[l+2]})$$ 
_with skip connection_ 
$$a^{[l+2]}=g(z^{[l+2]}+a^{[l]})$$ 

#### Why do we need ResNet ?
![vanishing/exploding gradients](https://1.bp.blogspot.com/-MXB-rQR4Zog/V5BOnQ4eiRI/AAAAAAAAYPQ/lZ9oudu6-2Y9N7gVR_Q621R4yvKZcUYVgCLcB/s1600/Screen%2BShot%2B2016-07-21%2Bat%2B12.23.59%2BAM.jpg)
>Training very deep NN (> 100 layers) poses problem of vanishing/exploding gradients rendering the the network to be non-converging or unstable.
>ResNets help reduce training error and address vanishing/exploding gradients problem

ResNets have unique advantage to learn identity function very easily hence even if we are adding residual block to a big NN using L2 regularization & ReLu activation then
$$a^{[l+2]}=g(z^{[l+2]}+a^{[l]})$$ 
would converge to 
$$a^{[l+2]}=g(a^{[l]})$$ 
$$a^{[l+2]}=a^{[l]}$$ 

Note: _As we go deeper into the network the error gradient used to update weights may decay exponentially(if less than 1) or explode given the cummulative effect of deep layers_

---
#### Regularization: DropOut

![Overfitting](https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Ftiriasresearch%2Ffiles%2F2018%2F01%2FOverfit-e1515527440489.jpg)

>Problem of overfitting/high variance occuers when network tries to completely fit the training data without generalizing. 

Overfitting can be addressed with
1. Regularization 
  (a) Lasso (L1)
  (b) Ridge (L2)
  (c ) Dropout
  (d) Data augmentation
  (e) Early stopping
2. More training data
3. Different NN arch

#### What is Dropout ?

>Randomly shutting off neurons in each iteration of training on a set results in effect of spreading out weights.
>Each neuron is dropped off with some fixed probability (1-keep_prob) and kept with probability keep_prob.

![dropout](https://static.commonlounge.com/fp/600w/aOLPWvdc8ukd8GTFUhff2RtcA1520492906_kc)

#### How Dropout works ?
```html
<script keep_prob=0.7
d3 = np.random.rand(a3.shape[0],a3.shape[1]) > keep_prob #generate a matrix of true/false
a3 = np.multiply(a3,d3)  #shut down randomly (1-keep_prob) units
a3 /= keep_prob #nullifies the effect of dropout
</script>
```
>Without dropout neurons tend to develop dependency on each other leading to overfitting however by dropping neurons at random we are forcing each neuron to learn features without dependency.

#### Dropout essentials
* Dropout is not recommended to be used on test set since it adds noise to the prediction however one can still opt to use dropout on test set but needs to average over several runs
* Different keep_prob can be defined for each layer. 
* Downside of using dropout is that cost function J(w,b) wouldn't no longer be strictly defined hence gradient descent will be difficult to comprehend
