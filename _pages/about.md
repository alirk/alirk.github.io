---
permalink: /
title: "Ali Ramezani-Kebrya"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a Tenured Associate Professor in the [Department of Informatics](https://www.mn.uio.no/ifi/english/index.html) at the [University of Oslo (UiO)](https://www.uio.no/english/). Before joining UiO, I have been a Senior Scientific Collaborator at [EPFL](https://www.epfl.ch/en/), working in [Laboratory for Information and Inference Systems (LIONS)](https://www.epfl.ch/labs/lions/). Before joining LIONS, I was an [NSERC Postdoctoral Fellow](https://www.nserc-crsng.gc.ca/students-etudiants/pd-np/pdf-bp_eng.asp) at the [Vector Institute](https://vectorinstitute.ai/) in Toronto. I finished my Ph.D. at the [University of Toronto](https://www.utoronto.ca/). My Ph.D. research was focused on developing theory and practices for next generation large-scale distributed and heterogeneous networks.


I work in machine learning and study **scalability**, **robustness**, **privacy**, **generalization**, and **stability** aspects of machine learning algorithms. In particular, I am developing **highly scalable**, **privacy-preserving**, and **robust** algorithms to train very large models in a distributed manner. Our algorithms can be used in so-called *federated learning* settings, where a deep model is trained on data distributed among multiple owners who cannot necessarily share their data, e.g., due to privacy concerns, competition, or by law. I also study the design of the underlying architecture, e.g, **neural networks** over which a learning algorithm is applied, in particular, the fundamental question of *How much should we overparameterize a neural network?*

-----

Recent News
======

- 1/2023: I joined the Department of Informatics at the University of Oslo!
- 12/2022: I gave a talk titled `Randomization Improves Deep Learning Security?` at the Annual Workshop of the VILLUM Investigator Grant at Aalborg University. 
- 10/2022: Our paper *[MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks](https://openreview.net/pdf?id=tqDhrbKJLS)* has been published in **Transactions on Machine Learning Research**. 
- 8/2022: I gave a talk titled `How Did DL Dominate Today’s ML? What Challenges and Limitations Remain?` at the University of Oslo.
- 6/2022: I gave a talk titled `Scalable ML: Communication-efficiency, Security, and Architecture Design` at the University of Edinburgh.
- 2/2022: I gave a talk titled `Scalable ML: Communication-efficiency, Security, and Architecture Design` at the University of Liverpool.
- 09/2021: Our paper *[Subquadratic Overparameterization for Shallow Neural Networks](https://proceedings.neurips.cc/paper/2021/hash/5d9e4a04afb9f3608ccc76c1ffa7573e-Abstract.html)* has been accepted to **NeurIPS 2021**.

-----

Selected Publications
======


<img style="float: left;" src="/images/MixTailor_Overview.png" width="350"/>   ML models are vulnerable to various attacks at training and test time including data/model poisoning and adversarial examples. We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed.  **MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks** increases computational complexity of designing tailored attacks for an informed adversary.

 
Ali Ramezani-Kebrya\*, Iman Tabrizian\*, Fartash Faghri, Ilya Markov, and Petar Popovski, **MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**, Transactions on Machine Learning Research, Oct. 2022.  
[pdf](https://openreview.net/pdf?id=tqDhrbKJLS){: .btn--research} [bib](https://www.jmlr.org/tmlr/papers/bib/tqDhrbKJLS.bib){: .btn--research} [code](https://github.com/Tabrizian/mix-tailor){: .btn--research} [arXiv](https://arxiv.org/abs/2207.07941){: .btn--research} [openreview](https://openreview.net/forum?id=tqDhrbKJLS){: .btn--research} 



<img style="float: left;" src="/images/OPnn.png" width="300"/>   Overparameterization refers to the important phenomenon where the width of a neural network is chosen such that learning algorithms can provably attain zero loss in nonconvex training. In **Subquadratic Overparameterization for Shallow Neural Networks**, we achieve the best known bounds on the number of parameters that is sufficient for gradient descent to converge to a global minimum with linear rate and probability approaching to one.

 
Chaehwan Song\*, Ali Ramezani-Kebrya\*, Thomas Pethick, Armin Eftekhari, and Volkan Cevher, **Subquadratic Overparameterization for Shallow Neural Networks**, NeurIPS 2021.  
[pdf](https://proceedings.neurips.cc/paper/2021/file/5d9e4a04afb9f3608ccc76c1ffa7573e-Paper.pdf){: .btn--research} [bib](https://scholar.googleusercontent.com/scholar.bib?q=info:kx3LBH3jDHQJ:scholar.google.com/&output=citation&scisdr=CgVA45jvEKCS5DjU-u4:AAGBfm0AAAAAY6XS4u5LcAezF3eXi9jM_VkuZB9hzmc2&scisig=AAGBfm0AAAAAY6XS4r70vs2W1lznoTFxd4JHdJ9kVKaF&scisf=4&ct=citation&cd=-1&hl=en){: .btn--research} [code](https://github.com/LIONS-EPFL/Subquadratic-Overparameterization){: .btn--research} [arXiv](https://arxiv.org/abs/2111.01875){: .btn--research} [openreview](https://openreview.net/forum?id=NhbFhfM960){: .btn--research} 


<img style="float: left;" src="/images/Result50_bs256.png" width="350"/>   In training deep models over multiple GPUs, the communication time required to share huge stochastic gradients is the main performance bottleneck. We closed the gap between theory and practice of unbiased gradient compression. **NUQSGD** is currently the method offering the highest communication-compression while still converging under regular (uncompressed) hyperparameter values.

 
Ali Ramezani-Kebrya, Fartash Faghri, Ilya Markov, Vitalii Aksenov, Dan Alistarh, and Daniel M. Roy, **NUQSGD: Provably Communication-Efficient Data-Parallel SGD via Nonuniform Quantization**, Journal of Machine Learning Research, vol. 22, pp. 1-43, Apr. 2021.  
[pdf](https://jmlr.org/papers/volume22/20-255/20-255.pdf){: .btn--research} [bib](https://www.jmlr.org/papers/v22/20-255.bib){: .btn--research} [code](https://github.com/fartashf/nuqsgd){: .btn--research} [arXiv](https://arxiv.org/abs/1908.06077){: .btn--research} 


<img style="float: left;" src="/images/MultiGPU.png" width="350"/>   Communication-efficient variants of SGD are often heuristic and fixed over the course of training. In **Adaptive Gradient Quantization for Data-Parallel SGD**, we empirically observe that the statistics of gradients of deep models change during the training and introduce two adaptive quantization schemes. We improve the validation accuracy by almost 2% on CIFAR-10 and 1% on ImageNet in challenging low-cost communication setups.  

 
Fartash Faghri\*, Iman Tabrizian\*, Ilya Markov, Dan Alistarh, Daniel M. Roy, and Ali Ramezani-Kebrya, **Adaptive Gradient Quantization for Data-Parallel SGD**, NeurIPS 2020.  
[pdf](https://papers.nips.cc/paper/2020/file/20b5e1cf8694af7a3c1ba4a87f073021-Paper.pdf){: .btn--research} [bib](https://scholar.googleusercontent.com/scholar.bib?q=info:xpAwoNIuzxUJ:scholar.google.com/&output=citation&scisdr=CgVA45jvEKCS5Djck8o:AAGBfm0AAAAAY6Xai8rnX4Rz-Zrxs7QCv6ocvm8RxOKV&scisig=AAGBfm0AAAAAY6Xaiy39O8cDh_0XnYOezqMyusWtK5Cu&scisf=4&ct=citation&cd=-1&hl=en){: .btn--research} [code](https://github.com/tabrizian/learning-to-quantize){: .btn--research} [arXiv](https://arxiv.org/abs/2010.12460){: .btn--research}

-----

Students 
======

- Co-supervision in ML-related Projects   
	- Thomas Michaelsen Pethick, Ph.D. in progress, EPFL.
	- Igor Krawczuk, Ph.D. in progress, EPFL.
	- Fabian Latorre, Ph.D. in progress, EPFL.
	- Wanyun Xie, research assistant, EPFL.
	- Ioannis Mavrothalassitis, master in progress, EPFL.
	- Xiangcheng Cao, master in progress, EPFL.
	- Seydou Fadel Mamar, master in progress, EPFL.	
	- Mohammadamin Sharifi, summer intern, EPFL.
	- Fartash Faghri, Ph.D. UoT,  first job after graduation: research scientist at Apple.
	- Iman Tabrizian, M.A.Sc. UoT,  first job after graduation: full-time engineer at NVIDIA.
	
-----	

Collaborators  and Friends of the Lab
======

- [Prof. Dan Alistarh](https://people.csail.mit.edu/alistarh/)
- [Dr. Kimon Antonakopoulos](https://people.epfl.ch/kimon.antonakopoulos?lang=en)
- [Prof. Volkan Cevher](https://people.epfl.ch/volkan.cevher?lang=en)
- [Dr. Grigorios Chrysos](https://people.epfl.ch/grigorios.chrysos?lang=en)
- [Prof. Min Dong](https://sites.google.com/ontariotechu.net/dong?pli=1) 
- [Prof. Ben Liang](https://www.comm.utoronto.ca/~liang/) 
- [Dr. Fanghui Liu](https://people.epfl.ch/fanghui.liu?lang=en)
- [Prof. Petar Popovski](http://petarpopovski.es.aau.dk/)
- [Prof. Daniel M. Roy](http://danroy.org/)




