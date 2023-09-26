# Visual Transformer

These files were created during my journey of learning PyTorch by taking inspiration from a PyTorch tutorial as well as the original ViT paper.
The tutorial can be found (https://www.learnpytorch.io/08_pytorch_paper_replicating/)[here] while the ViT paper is at this (https://arxiv.org/abs/2010.11929)[link].

The structure of the project is the following:
- main.py: the file to run to train from scratch a ViT architecture
- model.py: the classes used for creating the blocks needed by the ViT
- engine.py: training method returning also the stats of the training itself
- data_setup.py: utils functions for getting the dataloader in the correct format given the data paths
- **trials.ipynb**: some scratch notes and explanations about the process used in order to come up with the solution
- **finetuning.ipynb**: finetuning a pre-trained ViT model to obtain better performance on the task at hand

<p align='center'>
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png">
</p>
