# TMBuD Dataset 

Date created: Febr-2020

The Timisoara Building Dataset - TMBuD - is composed of 1120 images with the resolution of 768x1024 pixels. Our motivation for this is the belief that this resolution is a good balance between the processing resources needed for manipulating the image and the actual resolution of pictures made with smart devices. Moreover, this is the actual video resolution for filming using a smartphone, the main sensor for building detection systems.

TMBuD is created from images of buildings in Timisoara. Each building is presented from several perspectives, so this dataset can be used for evaluatinga building detection algorithm too. The dataset contains ground-truth imagesfor  salient  edges,  for  semantic  segmentation  and  the  GPS  coordinates  of  thebuildings. 

The STANDARD dataset contains 160 images grouped in the following sets: 100 consist of the training dataset, 25 consist of the validation data and 35 consistof the test data.

Total number of images are as following: 1120 images, 160 edge ground-truth images, 300 label ground-truth.

This is the CM(Multimedia Center) building Image Dataset, which we created from images of buildings in Timisoara. The dataset contains groundthruth images for salient edges and semantic segmentation of building. Please check with the authors of the TMBuD Dataset dataset, in case you are unsure about the respective copyrights and how they apply.

Please cite:

    @inproceedings{orhei2021tmbud,
    title={TMBuD: a dataset for urban scene building detection},
    author={Orhei, Ciprian and Vert, Silviu and Mocofan, Muguras and Vasiu, Radu},
    booktitle={International Conference on Information and Software Technologies},
    pages={251--262},
    year={2021},
    organization={Springer}
    }

To create the standard dataset sub-folders please run parse_database.py using _python parse_database.py --variant STANDARD. The files structure can be changed from _files.txt_.
Variants one can build:\
STANDARD - creates all 3 dataset of labels, edges and images\
BUILDING_DET_3 - creates a dataset of images where 3 images of each building are used for learning and at least 2 image for testing. The files are renamed to be similar tu ZuBuD dataset \
BUILDING_DET_3_NIGHT - creates a dataset of images where 3 images of each building are used for learning and at least 2 image day + 2 images night for testing. The files are renamed to be similar tu ZuBuD dataset \
BUILDING_DET_3_N - creates a dataset of images where 3 images of each building are used for learning and all the rest of images for testing. The files are renamed to be similar tu ZuBuD dataset \
SEMSEG_EVAL_FULL - creates a dataset of images with labeled ground truth into TRAIN (250 images) and TEST (50 images)\

Variants BUILDING_DET_3, BUILDING_DET_3_NIGHT, BUILDING_DET_3_N are still under development, so the number of images will keep increasing and mai change between TRAIN and TEST. 

The dataset contains 300 images grouped in the following sets:

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
