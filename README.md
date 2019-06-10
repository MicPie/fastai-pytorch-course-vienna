# Fastai v1 &amp; PyTorch v1 Course in Vienna
This is the repo for the [Fastai v1 &amp; PyTorch v1 Course in Vienna](https://keepcurrent.online/ml-course.html).
* [Fast.ai MOOC Details](https://www.fast.ai/2019/01/24/course-v3/)
* [Fast.ai MOOC Material](https://course.fast.ai) (this should be your first address if you are searching for something)
* [Fast.ai MOOC - Part 1 Notebooks](https://github.com/fastai/course-v3/tree/master/nbs/dl1)
* [Fastai docs](https://docs.fast.ai) (this should be your second address if you are searching for something)
* [fast.ai forum](https://forums.fast.ai) (this should be your third address if you are searching for something)
  * [Study group in Austria thread on the Fast.ai forum](https://forums.fast.ai/t/study-group-in-austria/26119)


## Dates
* 04.03.2019 - Stockwerk Coworking - Pater-Schwartz-Gasse 11A, 1150 Wien
* 18.03.2019 - Nic.at - Karlsplatz 1, 1010 Wien
* 01.04.2019 - EBCONT - Millennium Tower, Handelskai 94-96, 1200 Wien
* 15.04.2019 - EBCONT - Millennium Tower, Handelskai 94-96, 1200 Wien
* -- BREAK --
* 13.05.2019 - Nic.at - Karlsplatz 1, 1010 Wien
* 27.05.2019 - Wirtschaftskammer Österreich - Wiedner Hauptstraße 63, 1040 Wien
* 11.06.2019 - Nic.at - Karlsplatz 1, 1010 Wien

**[--> Please vote for the topics we should cover in detail in the last two lessons! <--](http://www.polljunkie.com/poll/xwfgmq/material-to-cover-in-the-last-2-fastai-pytorch-lessons)**


## Preparation
### Before the course
* [Install PyTorch and fastai on your machine](https://course.fast.ai/index.html) (see also [fastai developer installation](https://github.com/fastai/fastai#developer-install)).
  * [And/or set up external machine with GPU (see "Server setup")](https://course.fast.ai/) (without GPU the training will take much longer, so this is highly recommended).
* Install a Python IDE, e.g. [VS Code](https://code.visualstudio.com), for going through the fastai library in detail.
* Familiarize yourself with [Jupyter notebooks](https://github.com/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb), [terminal](https://course.fast.ai/terminal_tutorial.html), and the [Python debugger](https://github.com/nblock/pdb-cheatsheet/releases/download/v1.2/pdb-cheatsheet.pdf) (pdb.set_trace(), l, ll, u, n, c, etc.)
### During the course
* Fast.ai Lesson [Notes A](https://github.com/hiromis/notes) and [Notes B](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/)
* [Basic intro to deep learning](https://www.deeplearningbook.org/contents/intro.html)
* [Python learning resources (for Beginners and advanced)](https://forums.fast.ai/t/recommended-python-learning-resources/26888)
* Collection of [Python tips](http://book.pythontips.com/en/latest/)
* Basic matrix calculus:
  * [Matrix multiplication on German Wikipedia](https://de.wikipedia.org/wiki/Matrizenmultiplikation) (the German version has  better visualisations)
  * [Animated matrix multiplication](http://matrixmultiplication.xyz)
* [Broadcasting visualisation](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)

### Communication
There are several communication options:
* Fast.ai forums - https://forums.fast.ai/t/study-group-in-austria/26119 (preferred option)
* Slack - either on [Keep-Current (#mlcourse)](https://join.slack.com/t/keep-current/shared_invite/enQtMzY3Mzk1NjE2MzIzLWZlZWFjMDM1YWYxYmI5ZWE4YmZjNWYzMmNjMzlhMDYzOTIxZDViODhmNTMzZDI0NThmZWVlOTRjNjczZGJiOWE) or Vienna Data Science Group
* Facebook groups: [Keep-Current](https://www.facebook.com/groups/308893846340861/) or [Facebook Developer Circles](https://www.facebook.com/groups/DevCVienna/)



## Lesson 1 - Intro to fastai and PyTorch
* **To dos before the lesson:**
  * **watch the [fastai lesson 1](https://course.fast.ai/videos/?lesson=1) ([hiromis notes lesson 1](https://github.com/hiromis/notes/blob/master/Lesson1.md))**
  * **run the [lesson 1 notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)**
* fastai lesson 1 discussion
* Deep learning in a nutshell:
  * data
  * neural network
  * loss function
  * optimizer
* PyTorch intro and building blocks:
  * [PyTorch Blitz intro](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [another intro to PyTorch](http://deeplizard.com/learn/video/iTKbyFh-7GM) ([tensors and autograd picture](https://github.com/pytorch/pytorch))
  * PyTorch workflow: data array -> torch tensor -> torch dataset -> torch dataloader -> network training
  * torch.Tensor & Co. ([notebook](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_1_Intro.ipynb), based on [Part 1](http://deeplizard.com/learn/video/jexkKugTg04) and [Part 2](http://deeplizard.com/learn/video/AglLTlms7HU), [Broadcasting](https://pytorch.org/docs/master/notes/broadcasting.html))


## Lesson 2 - torch.nn & Co.
* **To dos before the lesson:**
  * **watch the [fastai lesson 2](https://course.fast.ai/videos/?lesson=2) ([hiromis notes lesson 2](https://github.com/hiromis/notes/blob/master/Lesson2.md))**
  * **run the [lesson 2 notebook about regression and SGD](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb)**
  * **try the [Udacity DL course exercise on PyTorch tensor](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%201%20-%20Tensors%20in%20PyTorch%20(Exercises).ipynb)**
  * **have a look at the [torch.nn tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html) to familiarize yourself with the concepts and to have it easier when we go through it in the meetup**
* PyTorch building blocks:
  * ([PyTorch Blitz intro](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html))
  * [EE-559 – Deep learning 1.6. Tensor internals](https://fleuret.org/ee559/ee559-slides-1-6-tensor-internals.pdf)
  * [torch.nn tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
  * [torch.nn notebook](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_2_torchnn.ipynb)
  * [torch.nn docs](https://pytorch.org/docs/stable/nn.html), incl. [nn.Parameter](https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter), weights, biases, gradients, etc.
  * [torch.nn.functional](https://pytorch.org/docs/stable/nn.html#torch-nn-functional)
  * [torch.optim](https://pytorch.org/docs/stable/optim.html)
  * [PyTorch and NumPy comparison notebook](https://github.com/odysseus0/pytorch_tutorials/blob/master/tensor_tutorial.ipynb)
* **A presentation about Cross entropy loss** by Christoph and Pascal
  * [Visual information theory](https://colah.github.io/posts/2015-09-Visual-Information/)


## Lesson 3 - Debugging, visualisation, and the fastai workflow
* **To dos before the lesson:**
  * **watch the [fastai lesson 3](https://course.fast.ai/videos/?lesson=3) ([hiromis notes lesson 3](https://github.com/hiromis/notes/blob/master/Lesson3.md))**
  * **run/have a look at the [lesson 3 notebook about multi-label prediction with Planet Amazon dataset](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)**
  * **try the [Udacity DL course exercise on Neural networks with PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%202%20-%20Neural%20Networks%20in%20PyTorch%20(Exercises).ipynb) [(solutions)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%202%20-%20Neural%20Networks%20in%20PyTorch%20(Solution).ipynb)**
  * **try the [Udacity DL course exercise on Neural network weight initialization with PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/weight-initialization/weight_initialization_exercise.ipynb) [(solutions)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/weight-initialization/weight_initialization_solution.ipynb)**
* PyTorch debugging:
  * [CMU DL debugging and visualisation presentation](http://deeplearning.cs.cmu.edu/recitations.spring19/slides_debugging.pdf) (optional [notebook](https://github.com/cmudeeplearning11785/Spring2019_Tutorials/blob/master/recitation-4/DataVisualization.ipynb))
  * [Debugging Intro Notebook](https://github.com/MicPie/fastai-pytorch-course-vienna/blob/master/PyTorch_3_debugging.ipynb)
  * [More meaningful error messages on CUDA](https://lernapparat.de/debug-device-assert/)
* [Lesson 3 notebook about multi-label prediction with Planet Amazon dataset](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb)
* fastai workflow & building blocks:
  * [PyTorch Dataset](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset), [PyTorch DataLoader](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader), and [fastai DataBunch](https://docs.fast.ai/basic_data.html#DataBunch).
  * [DataBlock API docs](https://docs.fast.ai/data_block.html) and [DataBlock API sample](https://github.com/hiromis/notes/blob/master/Lesson3.md#data-block-api-examples-2356)
  * Looking into the fastai library with your IDE
  * [layers](https://github.com/fastai/fastai/blob/master/fastai/layers.py)


## Lesson 4 - NLP
* **To dos before the lesson:**
  * **watch the [fastai lesson 4](https://course.fast.ai/videos/?lesson=4) ([hiromis notes lesson 4](https://github.com/hiromis/notes/blob/master/Lesson4.md))**
  * **Run the [lesson 3 imdb notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)**
  * **Run the [lesson 4 Movie recommendation notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb)**
  * **Have a look at the [Convolution arithmetic animations](https://github.com/vdumoulin/conv_arithmetic) and the notebooks from the CNN building blocks section below.**
* Convolution Neural Network building blocks:
  * [Animation of a Convolution Neural Network at work](https://www.youtube.com/watch?v=f0t-OCG79-U)
  * [Convolution arithmetic animations](https://github.com/vdumoulin/conv_arithmetic)
  * [Udacity Notebook - Convolutional Layer](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/conv_visualization.ipynb)
  * [Udacity Notebook - CNN Filters](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/custom_filters.ipynb)
  * [Udacity Notebook - Maxpooling Layer](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/maxpooling_visualization.ipynb)
  * [CS230 DL: C4M2: Deep Convolutional Models (CNN architectures, 1x1 conv., etc.)](https://cs230.stanford.edu/files/C4M2.pdf)
  * [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
* Natural Language Processing:
  * NLP notebook - Building NLP architecture & pipeline
  * Presentation - [ABC NLP](https://docs.google.com/presentation/d/1HYLX8-q3BrCW0mgbouqMo-KvYThgY6QpqAClZZlqdQw/edit?usp=sharing)
  * [Writing code for NLP Research](https://docs.google.com/presentation/d/17NoJY2SnC2UMbVegaRCWA7Oca7UCZ3vHnMqBV4SUayc/edit)
  * [NLP code implementations in python](https://github.com/graykode/nlp-tutorial) - NLP Tutorial
* [Learning tips](https://github.com/MicPie/fastai-pytorch-course-vienna#learning-tips)


## Lesson 5 - SGD, embeddings & Co.
* **To dos before the lesson:**
  * **watch the [fastai lesson 5](https://course.fast.ai/videos/?lesson=5) ([hiromis notes lesson 5](https://github.com/hiromis/notes/blob/master/Lesson5.md))**
  * **Run the [lesson 4 tabular notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-tabular.ipynb)**
  * **Run the [lesson 5 MNIST SGD notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist.ipynb)**
  * **Run the [lesson 7 MNIST ResNet notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb)**
  * **Have a look at the [lesson 6 Pets revisited notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb)**
* Recap basic training & Co.:
  * [MNIST SGD notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist.ipynb)
  * [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/) (see "Visualization of algorithms" section for animations)
  * [torch.optim docs](https://pytorch.org/docs/stable/optim.html)
  * [MNIST ResNet notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb) (see [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) and [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187))
  * [3D loss surface with and without ResNet blocks](https://github.com/tomgoldstein/loss-landscape#visualizing-3d-loss-surface) (from [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913))
  * (*Optional:* [Pets revisited notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb))
* *Optional:* Understanding SGD, RMSProp, and Adam:
  * [Understanding Exponentially Weighted Averages](https://www.youtube.com/watch?v=NxTFlzBjS-4)
  * [Bias Correction of Exponentially Weighted Averages](https://www.youtube.com/watch?v=lWzo8CajF5s)
  * [Gradient Descent With Momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I)
  * [RMSProp](https://www.youtube.com/watch?v=_e-LFe_igno)
  * [Adam](https://www.youtube.com/watch?v=JXQT_vxqwIs) (Momentum + RMSprop)
  * [Link to the slides from the videos above](http://cs230.stanford.edu/files/C2M2.pdf)
* [Learning tips](https://github.com/MicPie/fastai-pytorch-course-vienna#learning-tips)


## Lesson 6 - RNN & Co.
* **To dos before the lesson:**
  * **watch the [fastai lesson 6](https://course.fast.ai/videos/?lesson=6) ([hiromis notes lesson 6](https://github.com/hiromis/notes/blob/master/Lesson6.md))**
  * **have a look at the [Understanding-LSTMs blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)**
  * **run the [fastai lesson 3 IMDB notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)**
* RNN & Co.
  * [Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  * [nn.RNN & Co.](https://pytorch.org/docs/stable/nn.html#torch.nn.RNN) ([a simple RNN illustrated](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#creating-the-network))
  * [CMU - 11-785 - Recitation 7	- Recurrent Neural Networks](http://deeplearning.cs.cmu.edu/recitations.spring19/RNN_Recitation.pdf)
  * [PyTorch simple RNN](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/time-series/Simple_RNN.ipynb)
  * [nn.Embedding](https://pytorch.org/docs/stable/nn.html#embedding) (see also [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737))
  * Language model pretraining shown in the [Human Numbers notebook from Hiromis notes](https://github.com/hiromis/notes/blob/master/Lesson7.md#human-numbers-14311)
  * [fastai lesson 7 Human Numbers notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-human-numbers.ipynb)
* Self-study material:
  * [PyTorch Character-Level LSTM](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks/char-rnn)
  * [PyTorch Sentiment Analysis RNN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn)
  * [word2vec](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings)
  * [Attention notebook 1](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/attention) & [Attention notebook 2](https://github.com/MicPie/pytorch/blob/master/attention.ipynb) (see also [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/#attentional-interfaces) and [CS224n slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf))



## Lesson 7 - Practical final lesson
* **To dos before the lesson:**
  * **watch the [fastai lesson 7](https://course.fast.ai/videos/?lesson=7) ([hiromis notes lesson 7](https://github.com/hiromis/notes/blob/master/Lesson7.md))**
  * **Prepare your data and your notebooks so we can use the last 2 h as efficiently as possible.**
* Datasets ideas:
  * Tabular data: 
    * Heart disease classification: https://www.kaggle.com/ronitf/heart-disease-uci
  * Recommendation System:
    * Create an Artificial Sommelier: https://www.kaggle.com/zynicide/wine-reviews
  * Time series:
    * Suicide prediction: https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
  * Computer Vision:
    * Malaria infection classification: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
    * Dog breed classification: https://www.kaggle.com/jessicali9530/stanford-dogs-dataset
    * Image segmentation - lunar - https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset
    * Fruits classification - https://www.kaggle.com/moltean/fruits
    * Detect the artist, based on the image: https://www.kaggle.com/ikarus777/best-artworks-of-all-time
  * NLP:
    * Sarcasm detection: https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
    * Open analysis: Predict startup success by media mentions & comments - https://www.kaggle.com/hacker-news/hacker-news
  * GAN:
    * Create realistic images of the moon: https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset 
* Open lesson for going through practical applications with the fastai library.
* What next, how to keep going, and [keep learning](https://github.com/MicPie/fastai-pytorch-course-vienna#learning-tips)!


## Stuff that did not fit into our schedule:
* GANs
  * [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
  * [PyTorch MNIST GAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist)
  * [PyTorch DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  * [fastai WGAN notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-wgan.ipynb)
  * [PyTorch Cycle GAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan)
  * And for the very curious see [DFL Wasserstein GAN](http://www.depthfirstlearning.com/2019/WassersteinGAN) (contact Michael if you want to tackle this together)


## General PyTorch Deep Learning ressources
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)
* [PyTorch Docs](https://pytorch.org/docs)
* [Udacity Deep Learning PyTorch Notebooks](https://github.com/udacity/deep-learning-v2-pytorch)
* [CMU Deep Learning Course](http://deeplearning.cs.cmu.edu)
* [CMU Deep Learning Course Recitation Repo](https://github.com/cmudeeplearning11785/Spring2019_Tutorials)
* [Deep Lizard PyTorch Tutorials](http://deeplizard.com/learn/video/v5cngxo4mIg)
* [EE-559 – EPFL – Deep Learning](https://fleuret.org/ee559/)
* [Pytorch torch.einsum](https://rockt.github.io/2018/04/30/einsum) (= the best way to get familiar with matrix calculus and einsum)
* [PyTorch under the hood](https://speakerdeck.com/perone/pytorch-under-the-hood)
* [Advanced PyTorch concepts with code](https://github.com/graykode/projects)


## Deep Learning
* [The deep learning book (Ian Goodfellow and Yoshua Bengio and Aaron Courville)](https://www.deeplearningbook.org)
* [Neural Networks and Deep Learning (Michael Nielson)](http://neuralnetworksanddeeplearning.com)
* [ML yearning (Andrew Ng)](https://www.mlyearning.org) (About how to structure Machine Learning projects.)
* CS 230 Deep Learning Cheatsheets:
  * [Convolutional Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
  * [Recurrent Neural Networks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
  * [Deep Learning Tips and Tricks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)
* [AI Transformation Playbook (Andrew Ng)](https://landing.ai/ai-transformation-playbook/) (A playbook to become a strong AI company.)


## Mathematics
* [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html)
* [Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra)
* https://www.3blue1brown.com


## Selected publications
* [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG network)](https://arxiv.org/abs/1409.1556)
* [Deep Residual Learning for Image Recognition (ResNet network)](https://arxiv.org/abs/1512.03385)
* [Network In Network (1x1 convolutions)](https://arxiv.org/abs/1312.4400)
* [Going deeper with convolutions (Inception network)](https://arxiv.org/abs/1409.4842)
* Everything on https://distill.pub and https://colah.github.io.


## Possible presentation topics
Present one topic with a general introduction and PyTorch code in a Jupyter notebook in approx. 10-20 min. Feel free to add the notebooks to this repo.
* Weight decay, L1-, and L2 regularization ([see weight decay vs. L2 regularization](https://bbabenko.github.io/weight-decay/))
* Drop out (see [chapter 7.12](https://www.deeplearningbook.org/contents/regularization.html))
* [(fastai) Data augmentation](https://github.com/kechan/FastaiPlayground/blob/master/Quick%20Tour%20of%20Data%20Augmentation.ipynb)
* CNN ([Conv Nets: A Modular Perspective](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/) and [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/), [Convolution arithmetic animations](https://github.com/vdumoulin/conv_arithmetic), and [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)), or [Advanced CNN Architectures](https://dvl.in.tum.de/teaching/adl4cv-ws18/) (Advanced Deep Learning for Computer vision - Munich University)
* ResNets & DenseNets (network architecture, [PyTorch model code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), [loss function shape](https://arxiv.org/pdf/1712.09913), etc.)
* 1x1 convolutions ([Network In Network](https://arxiv.org/abs/1312.4400))
* Batch norm ([Udacity Notebook](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/batch-norm/Batch_Normalization.ipynb), [Batch Normalization](https://arxiv.org/abs/1502.03167), [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604), and [Group Normalization](https://arxiv.org/abs/1803.08494))
* LSTM unit ([Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))
* Attention ([Notebook](https://github.com/MicPie/pytorch/blob/master/attention.ipynb))
* Cross entropy loss (based on this [introduction](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) and [information theory](https://colah.github.io/posts/2015-09-Visual-Information/)).
* [Mixed precision training](https://forums.fast.ai/t/mixed-precision-training/20720), [floating point arithmetics](https://en.wikipedia.org/wiki/Floating-point_arithmetic), and the [fastai callback](https://docs.fast.ai/callbacks.fp16.html).


## Possible projects
* Tensorboard visualisation with fastai callback using [TensorboardX](https://github.com/lanpa/tensorboardX), including 2D visualisations for CNNs ([see starter notebook](https://github.com/MicPie/fastai_course_v3/blob/master/TBLogger_v2.ipynb) and [fastai forum thread](https://forums.fast.ai/t/tensorboard-integration/38023/))
* [Histopathologic Cancer Detection on Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)


## Deployment
* [fast.ai deployment options](https://course.fast.ai/deployment_render.html)
* [Pythonanywhere](https://www.pythonanywhere.com/) - free to start
* [Render](https://render.com/) - free for static (HTML) sites, 5$ for python hosting
* [Heroku](https://www.heroku.com/) - free basic cloud account for up to 3 projects

## Learning tips
* [10 Top Ideas to Help Your Learning & 10 Pitfalls to Avoid in Your Learning](https://barbaraoakley.com/wp-content/uploads/2018/02/10-Top-Ideas-to-Help-Your-Learning-and-10-Pitfalls-1.pdf) (from the [Learning how to learn](https://www.coursera.org/learn/learning-how-to-learn) course)
* Use spaced repetition to memorize important concepts, APIs, and everything else:
  * [Short intro to the spacing effect](https://fs.blog/2018/12/spacing-effect/)
  * [(More detailed) Introduction to augmenting Long-term Memory](http://augmentingcognition.com/ltm.html) ([concept outlined for a mathematic example](http://cognitivemedium.com/srs-mathematics))
  * [Spaced repitition in detail](https://www.gwern.net/Spaced-repetition)
  * [Anki spaced repitition flashcard app](https://apps.ankiweb.net)
