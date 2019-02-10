# Fastai v1 &amp; PyTorch v1 Course in Vienna
This is the repo for the [Fastai v1 &amp; PyTorch v1 Course in Vienna](https://keepcurrent.online/ml-course.html).<br>
* [Fast.ai MOOC Details](https://www.fast.ai/2019/01/24/course-v3/)<br>
* [Fast.ai MOOC Material](https://course.fast.ai)<br><br>

**Lesson 1 - Intro to fastai and PyTorch:**
* Deep learning in a nutshell: linear + nonlinear building blocks, e.g., min((k * x + d), 0)
* Basic matrix calculus:
  * [Matrix multiplication on German Wikipedia](https://de.wikipedia.org/wiki/Matrizenmultiplikation) (German version has  better visualisations)
  * [Animated matrix multiplication](http://matrixmultiplication.xyz)
* PyTorch intro:
  * [Tensors and autograd picture](https://github.com/pytorch/pytorch)
  * PyTorch workflow: Data -> Tensor -> Dataset -> Dataloader -> Network Training
  * [PyTorch Blitz intro](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* PyTorch building blocks:
  * [torch.nn](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
  * torch.nn.functional
* PyTorch debugging:
  * [Python Debugger Cheatsheet](https://github.com/nblock/pdb-cheatsheet/releases/download/v1.2/pdb-cheatsheet.pdf) (pdb.set_trace(), l, ll, u, n, c, etc.)
  * [More meaningful error messages on CUDA](https://lernapparat.de/debug-device-assert/)
  * [PyTorch Debugger layer](https://docs.fast.ai/layers.html#Debugger)
* https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch
* fastai workflow
<br><br>

**Lesson 2 - ?:**
* ?
<br><br>

**Lesson 3 - ?:**
* ?
<br><br>

**Lesson 4 - ?:**
* ?
<br><br>

**Lesson 5 - ?:**
* ?
<br><br>

**Lesson 6 - ?:**
* ?
<br><br>

**Lesson 7 - ?:**
* ?
<br><br>

**General PyTorch Deep Learning ressources:**<br>
* [PyTorch Tutorials](https://pytorch.org/tutorials/)<br>
* [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)<br>
* [PyTorch Docs](https://pytorch.org/docs)<br>
* [Udacity Deep Learning PyTorch Notebooks](https://github.com/udacity/deep-learning-v2-pytorch)<br>
* [CMU Deep Learning Lecture Notes](http://deeplearning.cs.cmu.edu)<br>
* [CMU Deep Learning Recitation Repo](https://github.com/cmudeeplearning11785/Spring2019_Tutorials)<br>
* [Pytorch torch.einsum](https://rockt.github.io/2018/04/30/einsum) (= the best way to get familiar with matrix calculus and einsum)
<br><br>

**Deep Learning:**
* [THE deep learning book](https://www.deeplearningbook.org)
* ???
<br><br>

**Mathematics:**
* ???
<br><br>

**Selected publications:**
* [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG network)](https://arxiv.org/abs/1409.1556)
* [Deep Residual Learning for Image Recognition (ResNet network)](https://arxiv.org/abs/1512.03385)
* [Network In Network (1x1 convolutions)](https://arxiv.org/abs/1312.4400)
* [Going deeper with convolutions (Inception network)](https://arxiv.org/abs/1409.4842)
* ???
* Everything on https://distill.pub and https://colah.github.io.
<br><br>

**Possible presentation topics:**
* CNN ([Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/) and [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/))
* ResNets & DenseNets (theory, PyTorch code, loss function shape)
* 1x1 convolutions
* LSTM unit ([Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))
* Cross entropy loss (based on this [introduction](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) and [information theory](https://colah.github.io/posts/2015-09-Visual-Information/)).
<br><br>

**Possible projects:**
* Tensorboard visualisation with fastai callback using [TensorboardX](https://github.com/lanpa/tensorboardX), including 2D visualisations for CNNs ([see notebook](https://github.com/MicPie/fastai_course_v3/blob/master/TBLogger_v2.ipynb))
* [Histopathologic Cancer Detection on Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)
<br><br>

**Learning tips:**
* [10 Top Ideas to Help Your Learning & 10 Pitfalls to Avoid in Your Learning](https://barbaraoakley.com/wp-content/uploads/2018/02/10-Top-Ideas-to-Help-Your-Learning-and-10-Pitfalls-1.pdf) (from the [Learning how to learn](https://www.coursera.org/learn/learning-how-to-learn) course)
* Use spaced repetition to memorize important concepts, APIs, and everything else:
  * [Introduction to augmenting Long-term Memory](http://augmentingcognition.com/ltm.html)
  * [Spaced repitition in detail](https://www.gwern.net/Spaced-repetition)
  * [Anki spaced repitition flashcard app](https://apps.ankiweb.net)
