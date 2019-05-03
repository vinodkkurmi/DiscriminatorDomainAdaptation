# Looking back at Labels: A Class based Domain Adaptation Technique(IDDA)

Torch code for Domain Adaptation model(IDDA) . For more information, please refer the [paper](https://arxiv.org/abs/1904.01341) 

Accepted at [[IJCNN 2019 Oral](https://www.ijcnn.org/)]

#####  [[Project  Page Link ]](https://vinodkkurmi.github.io/DiscriminatorDomainAdaptation/)     [[Paper Link ]](https://arxiv.org/pdf/1904.01341.pdf)

#### Abstract 
In this paper, we tackle a problem of Domain Adaptation. In a domain adaptation setting, there is provided a labeled set of examples in a source dataset with multiple classes being present and a target dataset that has no supervision. In this setting, we propose an adversarial discriminator based approach. While the approach based on adversarial discriminator has been previously proposed; in this paper, we present an informed adversarial discriminator. Our observation relies on the analysis that shows that if the discriminator has access to all the information available including the class structure present in the source dataset, then it can guide the transformation of features of the target set of classes to a more structured adapted space. Using this formulation, we obtain the state-of-the-art results for the standard evaluation on benchmark datasets. We further provide detailed analysis which shows that using all the labeled information results in an improved domain adaptation.

![Result](http://home.iitk.ac.in/~vinodkk/idda_model/idda_model.png) 

### Requirements
This code is written in Lua and requires [Torch](http://torch.ch/). 


You also need to install the following package in order to sucessfully run the code.
- [Torch](http://torch.ch/docs/getting-started.html#_)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)


#### Download Dataset
- [Office -31](https://pan.baidu.com/s/1o8igXT4)
- [ImageClef](https://pan.baidu.com/s/1lx2u1SMlSamsHnAPWrAHWA)
- [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)

##### Prepare Datasets
- Download the dataset


### Training Steps

We have prepared everything for you ;)

####Clone the repositotry 

``` git clone https://github.com/vinodkkurmi/DiscriminatorDomainAdaptation  ```

#### Dataset prepare
- Downalod dataset

-  put all source images inside mydataset/train/ such that folder name is class name
```
  mkdir -p /path_to_wherever_you_want/mydataset/train/ 
```
- put all target images inside mydataset/val/ such that folder name is class name

``` 
mkdir -p /path_to_wherever_you_want/mydataset/val/ 
```
- creare softlink of dataset
```
 cd DiscriminatorDomainAdaptation/
 ln -sf /path_to_wherever_you_want/mydataset dataset
```
 
  

#### Pretrained Alexnet model
- Download Alexnet pretraine caffe model [Link](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

``` 
cd DiscriminatorDomainAdaptation/  
```

```
ln -sf /path_to_where_model_is_downloaded/ pretrained_network 
```

#### Train model
``` 
cd DiscriminatorDomainAdaptation/  
./train.sh 
```




### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@article{kurmi2019looking,
  title={Looking back at Labels: A Class based Domain Adaptation Technique},
  author={Kurmi, Vinod Kumar and Namboodiri, Vinay P},
  journal={arXiv preprint arXiv:1904.01341},
  year={2019}
}
```

## Contributors
* [Vinod K. Kurmi][1] (vinodkk@iitk.ac.in)



[1]: https://github.com/vinodkkurmi




