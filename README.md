# DistillationTraining
Distillation Knowledge for training Multi-exit Model
We implement a Distillation knowledge to train a Multi-Exit (ME) Model ResNet50. 
Our ME consists of 4 early exit gates from each residual block. Using this schenario, we yield 82.5% 85% 89% 92% accuracy from gate 1 to 4 respectively.
To reproduce the result, run the [main.py](https://github.com/nadeny/DistillationTraining/blob/main/main.py) script.

![alt text](https://i.imgur.com/iQpGeAm.png)

Reference:

[Distillation-Based Training for Multi-Exit Architectures](https://openaccess.thecvf.com/content_ICCV_2019/papers/Phuong_Distillation-Based_Training_for_Multi-Exit_Architectures_ICCV_2019_paper.pdf)
