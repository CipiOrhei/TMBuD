# TMBuD Dataset 

Date created: Febr-2020

The Timisoara Building Dataset - TMBuD - is composed of 160 images with the resolution of 768x1024 pixels. Our motivation for this is the belief that this resolution is a good balance between the processing resources needed for manipulating the image and the actual resolution of pictures made with smart devices. Moreover, this is the actual video resolution for filming using a smartphone, the main sensor for building detection systems.

TMBuD is created from images of buildings in Timisoara. Each building ispresented from several perspectives, so this dataset can be used for evaluatinga building detection algorithm too. The dataset contains ground-truth imagesfor  salient  edges,  for  semantic  segmentation  and  the  GPS  coordinates  of  thebuildings. 

The standard dataset contains 160 images grouped in the following sets: 100 consist of the training dataset, 25 consist of the validation data and 35 consistof the test data. 

This is the CM(Multimedia Center) building Image Dataset, which we created from images of buildings in Timisoara. The dataset contains groundthruth images for salient edges and semantic segmentation of building. Please check with the authors of the CM Building Dataset dataset, in case you are unsure about the respective copyrights and how they apply.

Please cite:

    @inproceedings{orhei2021tmbud,
    title={TMBuD: a dataset for urban scene building detection},
    author={Orhei, Ciprian and Vert, Silviu and Mocofan, Muguras and Vasiu, Radu},
    booktitle={International Conference on Information and Software Technologies},
    pages={251--262},
    year={2021},
    organization={Springer}
    }

To create the standard dataset sub-folders please run parse_database.py using _python parse_database.py --variant STANDARD_. The files structure can be changed from _files.txt_.

The dataset contains 160 images grouped in the following sets:

CLASSES :\
BACKGROUND	=	0	=	(	0,		0,		0) \
BUILDING	  =	1	=	(	125,	125,	0)\
DOOR		    =	2	=	(	0,		125,	125)\
WINDOW		  =	3	=	(	0,		255,	255)\
SKY			    =	4	=	(	255,	0,		0)\
VEGETATION	=	5	=	(	0,		255,	0)\
GROUND		  =	6	=	(	125,	125,	125)\
NOISE		    =	7	=	(	0,		0,		255)


![image](https://user-images.githubusercontent.com/77099016/111601436-0536d800-87db-11eb-93f8-20dcb25d9a9e.png)
