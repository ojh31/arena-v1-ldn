


What won't be covered today:

>Quite a lot, including Megatron and Deepspeed





## Theory that would be good to read in advance

[Make ML go brrr](https://horace.io/brrr_intro.html)

[How do GPUs really work?](https://lilianweng.github.io/posts/2021-09-25-train-large/)





## Torch run on single GPU in colab

In Colab, you will get 12 hours of execution time but the session will be disconnected if you are idle for more than 60 minutes.

Our example uses mnist training py script for single GPU [from here](https://github.com/pytorch/examples/blob/main/mnist/main.py)

Provide code in colab notebook: walk through it and highlight how changing the device changes GPU use

This is my notebook: https://colab.research.google.com/drive/1RyltDeb63NzqBsaw1eqVdmPOl7O25o3J?usp=sharing

Running 1 epoch should take about 25 seconds vs 2 mins 30 seconds, if you have a Tesla T4 GPU




## Overview of parallelism options

Data parallelism: datasets are broken into subsets which are processed in batches on different GPUs using the same model. The results are then combined and averaged in one version of the model. This method relies on the DataParallel class.

Distributed data parallelism: enables you to perform data parallelism across GPUs and physical machines and can be combined with model parallelism. This method relies on the DistributedDataParallel class.

Model parallelism: a single model is broken into segments with each segment run on different GPUs. The results from the segments are then combined to produce a completed model. 

Model parallelism can break the model apart by layers (often 'pipeline parallelism') or distribute within each layer ('tensor parallelism'). 

Lots more on this: 





## Setting up Tensorboard profiler with pyTorch

!!!### to add something on this next week

https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html





## Data parallelism

Define and save a pytorch model

Set device according to local_rank number

Slower than distributeddataparallel, so we won't do it

different pytorch parallel options: https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained#:~:text=To%20use%20data%20parallelism%20with,object%20with%20a%20DataParallel%20object.&text=Then%2C%20when%20you%20call%20your,distributed%20across%20your%20defined%20GPUs.




## Cloud provider options

Main ones (AWS, GCP, Azure) provide huge range of services, inc *much more expensive* GPU compute! 

Lambda is cheaper. Vast is cheaper still. 

Other providers have other advantages: e.g.: CoreWeave has large clusters of GPUs with high bandwidth connections.

There are many other cloud providers not mentioned here!





## Getting started with Lambda Labs

###!!! Add place to put ssh key ssh1.pem which lambda gives you

if you want to attach a filesystem:
1. make the filesystem before provisioning the machine. 
2. ensure the filesystem and machine are in the same region

Otherwise just provision as before. 



## Vast.ai

Vast.ai is cheaper that other providers: maybe half the price of lambda

So you may wish to use it for training large models if paying with your own money. 

But also, you're renting GPUs from who-knows-who. Which may be fine for learning or personal projects. Questionable if it's the best choice within an organisation. 

Vast also doesn't include a storage system: just the disk of the server.

You'll need to install packages with Vast unless you pick a docker image with everything you need. 

To download data onto it you may wish to use wormhole CLI tool for the transfer.

If you provision a machine don't forget to increase the attached disk size! It gives you 500mb by default. 

You can also set the Docker image to base your machine on (ie, it comes ith CUDA installed to save you installing it all from scratch). E.g.: nvidia/cuda:11.1.1-devel-ubuntu20.04 






## Distributed data parallel

Does this store a copy of the model on each GPU or each node? 

Does it average all the weights every now and then? 

Does it do more training per epoch if more GPUs?

For a single node with many GPUs this was the key: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py

For multiple nodes would use slurm or k8s to make a cluster of nodes and run jobs on that: https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series








## Exercises

Pick a model, any model. Get it training on a single GPU

Run training on a single node with multiple GPUs. If there aren't enough multi-GPU servers available for everyone, perhaps provision one and take it in turns: the actual training shouldn't take long. Or use [Vast.ai](https://vast.ai/). It's more about getting the setup right. [This may prove helpful](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series)

If you want to distribute training across lots of nodes, how might docker and kubernetes come in handy?

For bonus points:

>Set up pytorch monitoring dashboard to show GPU usage and training statistics






## This is very much the tip of the iceberg!

Other pytorch approaches to distributed training include:

1. Torchelastic

2. Fully shared data parallel (FSDP)

3. Megatron


























