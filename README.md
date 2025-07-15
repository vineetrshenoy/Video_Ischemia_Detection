# Video-based method for Ischemia Detection in Hand Trauma

This repository contains the code associated with the paper
```
Perfusion Assessment of Healthy and Injured Hands Using Video-Based Deep Learning Models
Shenoy V., Kingston C., Singh M., Durr N., Chellappa R., Giladi A.
Plastic and Reconstructive Surgery 2025
```

## Running the Code

Unfortunately, due to IRB restrictions we can not releaes the data. However, you can use this code to train your own model. To train the iPPG extractor, run the following command

```
python main.py --config-file config/cross_validation.yaml
```

To train the perfusion classifier, use the following command
```
python main_classifier.py --config-file config/classifier_training.yaml
```