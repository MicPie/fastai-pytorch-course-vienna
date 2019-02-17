# Fastai v1 &amp; PyTorch v1 Course in Vienna
This is the repo for the [Fastai v1 &amp; PyTorch v1 Course in Vienna](https://keepcurrent.online/ml-course.html).
* [Fast.ai MOOC Details](https://www.fast.ai/2019/01/24/course-v3/)<br>
* [Fast.ai MOOC Material](https://course.fast.ai)<br>
* [Study group in Austria thread on the Fast.ai forum](https://forums.fast.ai/t/study-group-in-austria/26119)


## Preparation:
### Before the course:
* [Install PyTorch and fastai on your machine](https://course.fast.ai/index.html) (see also [fastai developer installation](https://github.com/fastai/fastai#developer-install)).
  * [And/or set up external machine with GPU (see "Server setup")](https://course.fast.ai/) (without GPU the training will take much longer, so this is highly recommended).
* Install a Python IDE, e.g. [VS Code](https://code.visualstudio.com), for going through the fastai library in detail.
* Familiarize yourself with the [Python Debugger Cheatsheet](https://github.com/nblock/pdb-cheatsheet/releases/download/v1.2/pdb-cheatsheet.pdf) (pdb.set_trace(), l, ll, u, n, c, etc.)
### During the course:
* [Basic intro to deep learning](https://www.deeplearningbook.org/contents/intro.html)
* [Python learning resources (for Beginners and advanced)](https://forums.fast.ai/t/recommended-python-learning-resources/26888)
* Basic matrix calculus:
  * [Matrix multiplication on German Wikipedia](https://de.wikipedia.org/wiki/Matrizenmultiplikation) (the German version has  better visualisations)
  * [Animated matrix multiplication](http://matrixmultiplication.xyz)
* [Broadcasting visualisation](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)


## Lesson 1 - Intro to fastai and PyTorch:
* Deep learning in a nutshell: linear + nonlinear building blocks, e.g., max((k * x + d), 0); general function approximation
* PyTorch intro and building blocks:
  * [Intro to PyTorch](http://deeplizard.com/learn/video/iTKbyFh-7GM)
  * [Tensors and autograd picture](https://github.com/pytorch/pytorch)
  * PyTorch workflow: data array -> torch tensor -> torch dataset -> torch dataloader -> network training
  * torch.Tensor & Co. ([Notebook](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_1_Intro.ipynb), based on [Part 1](http://deeplizard.com/learn/video/jexkKugTg04) and [Part 2](http://deeplizard.com/learn/video/AglLTlms7HU), [Broadcasting](https://pytorch.org/docs/master/notes/broadcasting.html))
  * [torch.nn](https://pytorch.org/tutorials/beginner/nn_tutorial.html) incl. weights, biases, gradients, etc.
  * torch.nn.functional
  * torch.optim
  * [PyTorch Blitz intro](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)


## Lesson 2 - Debugging and visualisation:
* PyTorch debugging:
  * [CMU DL debugging and visualisation](http://deeplearning.cs.cmu.edu/recitations.spring19/slides_debugging.pdf) incl. [Notebook](https://github.com/cmudeeplearning11785/Spring2019_Tutorials/blob/master/recitation-4/DataVisualization.ipynb)
  * [More meaningful error messages on CUDA](https://lernapparat.de/debug-device-assert/)
  * [PyTorch Debugger layer](https://docs.fast.ai/layers.html#Debugger)
* ?


## Lesson 3 - ?:
* https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch
* fastai workflow
* ?


## Lesson 4 - ?:
* ?


## Lesson 5 - ?:
* ?


## Lesson 6 - ?:
* ?


## Lesson 7 - ?:
* ?


## General PyTorch Deep Learning ressources:
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
* [PyTorch Docs](https://pytorch.org/docs)
* [Udacity Deep Learning PyTorch Notebooks](https://github.com/udacity/deep-learning-v2-pytorch)
* [CMU Deep Learning Course](http://deeplearning.cs.cmu.edu)
* [CMU Deep Learning Course Recitation Repo](https://github.com/cmudeeplearning11785/Spring2019_Tutorials)
* [Deep Lizard PyTorch Tutorials](http://deeplizard.com/learn/video/v5cngxo4mIg)
* [Pytorch torch.einsum](https://rockt.github.io/2018/04/30/einsum) (= the best way to get familiar with matrix calculus and einsum)


## Deep Learning:
* [THE deep learning book](https://www.deeplearningbook.org)


## Mathematics:
* https://www.3blue1brown.com


## Selected publications:
* [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG network)](https://arxiv.org/abs/1409.1556)
* [Deep Residual Learning for Image Recognition (ResNet network)](https://arxiv.org/abs/1512.03385)
* [Network In Network (1x1 convolutions)](https://arxiv.org/abs/1312.4400)
* [Going deeper with convolutions (Inception network)](https://arxiv.org/abs/1409.4842)
* Everything on https://distill.pub and https://colah.github.io.


## Possible presentation topics:
* Weight decay (L1 vs. L2, loss function vs. real weight decay)
* Drop out
* Data augmentation
* CNN ([Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/) and [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), [Convolution arithmetic animations](https://github.com/vdumoulin/conv_arithmetic), and [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/))
* ResNets & DenseNets (theory, PyTorch code, loss function shape, etc.)
* 1x1 convolutions ([Network In Network](https://arxiv.org/abs/1312.4400))
* Batch norm ([Udacity Notebook](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/batch-norm/Batch_Normalization.ipynb), [Batch Normalization](https://arxiv.org/abs/1502.03167), [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604), and [Group Normalization](https://arxiv.org/abs/1803.08494))
* LSTM unit ([Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))
* Attention ([Notebook](https://github.com/MicPie/pytorch/blob/master/attention.ipynb))
* Cross entropy loss (based on this [introduction](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) and [information theory](https://colah.github.io/posts/2015-09-Visual-Information/)).


## Possible projects:
* Tensorboard visualisation with fastai callback using [TensorboardX](https://github.com/lanpa/tensorboardX), including 2D visualisations for CNNs ([see starter notebook](https://github.com/MicPie/fastai_course_v3/blob/master/TBLogger_v2.ipynb) and [fastai forum thread](https://forums.fast.ai/t/tensorboard-integration/38023/))
* [Histopathologic Cancer Detection on Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)


## Learning tips:
* [10 Top Ideas to Help Your Learning & 10 Pitfalls to Avoid in Your Learning](https://barbaraoakley.com/wp-content/uploads/2018/02/10-Top-Ideas-to-Help-Your-Learning-and-10-Pitfalls-1.pdf) (from the [Learning how to learn](https://www.coursera.org/learn/learning-how-to-learn) course)
* Use spaced repetition to memorize important concepts, APIs, and everything else:
  * [Introduction to augmenting Long-term Memory](http://augmentingcognition.com/ltm.html)
  * [Spaced repitition in detail](https://www.gwern.net/Spaced-repetition)
  * [Anki spaced repitition flashcard app](https://apps.ankiweb.net)
