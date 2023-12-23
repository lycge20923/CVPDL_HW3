# HW3 Report 

## Report 
### 1.Image Captioning
#### a.Compare the performance of different pre-trained models in generating captions

* I selected the all 4 pre-trained models in generating captions
  1. [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
  2. [Salesforce/blip2-opt-6.7b-coco](https://huggingface.co/Salesforce/blip2-opt-6.7b-coco)
  3. [Salesforce/blip2-opt-6.7b](https://huggingface.co/Salesforce/blip2-opt-6.7b)
  4. [Salesforce/blip-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
* How to evaluate which one of the four pre-trained models has the best performance? I used CLIP for evaluation. I loaded the CLIP Model with the pre-trained weight, ```ViT-B/32```. Afterward, for each image and its corresponding text, I performed preprocessing, encoded to obtain image and text features, and calculated the cosine similarity between them. In the last step, I summed up all cosine similarities and divided by the number of images to get the final score.
* After conducting ```CUDA_VISIBLE_DEVICES=0 python ImgCaption.py```, the final scores are shown below(Could also check ```./pretrained_compare_results.txt```):
  1. Salesforce/blip2-opt-2.7b:0.77609
  2. Salesforce/blip2-opt-6.7b-coco:0.7805
  3. Salesforce/blip2-opt-6.7b:0.7777
  4. Salesforce/blip-flan-t5-xl:0.7757
* The pre-trained weight,```Salesforce/blip2-opt-6.7b-coco```, has the best performance. Thus we used it to generate the image captions.

#### b.Design 2 templates of prompts for later generating comparison

* I designed three templates of prompts for later generating comparisons. All prompts have two parts: **predix** and **suffix**. The prefix among all prompts is the same: **a real photo of "Generated_Text", "label", width: 'width', height: 'height',  in the aquarium, undersea background, non-grayscale**. The difference lies in the suffix:
  1. First: **camera shake, low HD quality**
  2. Second: **high HD quality, limited color**
  3. Third: **camera shake, low HD quality, splashes, limited color**

* In this homework, we will generate image captions for two purposes: one is for calculating the FID Score, and the other is for data augmentation. For the first purpose, all images in the training dataset will be used to generate texts and prompts. For the second purpose, however, since it is recommended to use images with no more than 6 bounding boxes and containing only one category, I will additionally select images that meet these criteria.

### 2. Text-to-Image Generation
#### a. Generate Images from Text Grounding

* Since I created three kinds of prompts, in this problem I would used all prompts to generate images. The all results are stored in ```result_23```

### b. Generate Images from Image Grounding
* Same as ```a```, the results would be stored in ```result_23```

### 3. Compare FID

* Text Grounding

  * Since I created three kinds of prompts, in this problem I would instead compare the performance of all three prompts.The FID scores are stored in ```result_23``` and shown here:

    |Prompts|FID Score|
    |-------|---------|
    | First | 145.8596|
    | Second| 145.8691|
    | Third | 142.8163|

  * It could be shown that the differences weren't obvious. We could conclude that the description of the **species** and **background** is more significant than other details when generating images using **GLIGEN**. In the future, we could use more prompt types to verify our conjecture. 

* Image Grounding

  * We could find the **third** prompt has the lowest FID result .Therefore, we then used the **third** prompt to conduct image generating by image grounding.

  * The **FID result** is 144.8596. 
    |Prompts|FID Score|
    |-------|---------|
    | Third | 144.8596|
  
  * It is interesting that the FID result of image grounding isn't better than that of text grounding. I think the reason is that I didn't choose the image I intended to generate, which led to the result of the generation not improving.

### 4. Improvement of Detection Model

* Brief Description

  1. How to do Data Augmentation? Since the number of images containing fish is the largest, I then tried to generate a proper amount of images of other species so that the number of images from each species could be the same. In the end, there would be 1474 images in the dataset, which contains 448 original images and 1026 generated images.

  2. Under the recommendation for generating images (using images with no more than 6 bounding boxes and containing only one category), I suddenly have a question: Is it better to generate images with the same bounding boxes as the original images? Alternatively, should we randomly select the bounding boxes we want to generate (of course, within the recommended guidelines)? Therefore, I have trained the models with four datasets, the first two training datasets in which the generated images have the same bounding boxes as the images in the original dataset, while in the last two training datasets, the range of the bounding boxes of generated images is randomly selected.

* The MaP Results are shown:
  |Data Augment|Bouding Boxes|Grounding|MAP|
  |-|-------------|---------|---|
  |X|Original|X|0.4695|
  |V|Original|Text Groudning|0.4549|
  |V|Original|Image Groudning|0.4544|
  |V|Random|Text Groudning|0.4655|
  |V|Random|Image Groudning|0.4729|

## Excution

### Preliminary Action
#### 1. Download HW1 Dataset
1. Go to *NTU COOL*, download the ```hw1_dataset.zip``` and put it in the directory.
2. unzip the ```hw1_dataset.zip```: ``` unzip hw1_dataset.zip```

#### 2.Create a Virtual Environment

```
conda create --name your_name python=3.11 -y
source activate your_name
```

#### 3.Download the related packages

```
pip install numpy==1.23.5
pip install cython
pip install -r requirements.txt
pip uninstall -y torchtext
```

#### 4.Download ```GLIGEN``` and its relative pre-trained weights

* Download ```GLIGEN```
  ```
  git clone https://github.com/gligen/GLIGEN.git
  ```
* Download pre-trained weights
  ```
  wget https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin?download=true -O checkpoint_generation_text.pth
  wget https://huggingface.co/gligen/gligen-generation-text-image-box/resolve/main/diffusion_pytorch_model.bin?download=true -O checkpoint_generation_text_image.pth
  ```

### 5. Download ```DAB-DETR``` and its relative pre-trained weights

* Download ```DAB-DETR`` and setup.

  ```
  git clone https://github.com/IDEA-Research/DAB-DETR.git
  cd DAB-DETR/models/dab_deformable_detr/ops
  python setup.py build install
  # unit test (should see all checking is True)
  python test.py
  cd ../../../..
  ```

* Download the pre-trained weight: 
  1. Go to the website: https://drive.google.com/drive/folders/1vVxRu8wmNA7sxdEB46vcoYlXvJIy2-Ep
  2. Download the ```checkpoint.pth``` 
  3. Move the ```checkpoint.pth``` to the ```CVPDL_HW3``` directory.

### Run 
#### 1. Generate Image Caption

* Must add ```CUDA_VISIBLE_DEVICES=0``` in excution code, and change the **Device Number** to the GPU you could run.
* After running the following code, the results would all store in ```./result_1/```

  ```
  CUDA_VISIBLE_DEVICES=0 python 1.py 
  ```

### 2. Text-to-Image Generation
### 3. Calculate FID

* Copy the file ```23.py``` to the ```GLIGEN``` directory, move to it and execute.

  ```
  cp -r 23.py GLIGEN/23.py
  cd GLIGEN
  python 23.py --batch_size 1
  ```

### 4. Improvement Model
* To generate the needed images, running 
  ```
  cp -r 4-1.py GLIGEN/4-1.py
  cd GLIGEN
  python 4-1.py --batch_size 1
  ```

* After generating the images, we then prepared for the training datasets and annotations.

  1. Original Boxes(with Text Grounding)
    ```
    cp -r hw1_dataset result_4/originalboxes/hw1_dataset_TG
    mv result_4/originalboxes/hw1_dataset_TG/annotations/val.json result_4/originalboxes/hw1_dataset_TG/annotations/instances_val2017.json
    mv 'result_4/train_Text_Grounding(original).json' result_4/originalboxes/hw1_dataset_TG/annotations/instances_train2017.json
    mv result_4/originalboxes/hw1_dataset_TG/train result_4/originalboxes/hw1_dataset_TG/train2017
    mv result_4/originalboxes/hw1_dataset_TG/valid result_4/originalboxes/hw1_dataset_TG/val2017
    mv result_4/originalboxes/generation_box_text/* result_4/originalboxes/hw1_dataset_TG/train2017
    rm -r result_4/originalboxes/generation_box_text
    ```
  
  2. Original Boxes(with Image Grounding)
    ```
    cp -r hw1_dataset result_4/originalboxes/hw1_dataset_IG
    mv result_4/originalboxes/hw1_dataset_IG/annotations/val.json result_4/originalboxes/hw1_dataset_IG/annotations/instances_val2017.json
    mv 'result_4/train_Image_Grounding(original).json' result_4/originalboxes/hw1_dataset_IG/annotations/instances_train2017.json
    mv result_4/originalboxes/hw1_dataset_IG/train result_4/originalboxes/hw1_dataset_IG/train2017
    mv result_4/originalboxes/hw1_dataset_IG/valid result_4/originalboxes/hw1_dataset_IG/val2017
    mv result_4/originalboxes/generation_box_image/* result_4/originalboxes/hw1_dataset_IG/train2017
    rm -r result_4/originalboxes/generation_box_image
    ```  
  
  3. Random Boxes(with Text Grounding)
    ```
    cp -r hw1_dataset result_4/randomboxes/hw1_dataset_TG
    mv result_4/randomboxes/hw1_dataset_TG/annotations/val.json result_4/randomboxes/hw1_dataset_TG/annotations/instances_val2017.json
    mv 'result_4/train_Text_Grounding(random).json' result_4/randomboxes/hw1_dataset_TG/annotations/instances_train2017.json
    mv result_4/randomboxes/hw1_dataset_TG/train result_4/randomboxes/hw1_dataset_TG/train2017
    mv result_4/randomboxes/hw1_dataset_TG/valid result_4/randomboxes/hw1_dataset_TG/val2017
    mv result_4/randomboxes/generation_box_text/* result_4/randomboxes/hw1_dataset_TG/train2017
    rm -r result_4/randomboxes/generation_box_text
    ```
  
  4. Random Boxes(with Image Grounding)
    ```
    cp -r hw1_dataset result_4/randomboxes/hw1_dataset_IG
    mv result_4/randomboxes/hw1_dataset_IG/annotations/val.json result_4/randomboxes/hw1_dataset_IG/annotations/instances_val2017.json
    mv 'result_4/train_Image_Grounding(random).json' result_4/randomboxes/hw1_dataset_IG/annotations/instances_train2017.json
    mv result_4/randomboxes/hw1_dataset_IG/train result_4/randomboxes/hw1_dataset_IG/train2017
    mv result_4/randomboxes/hw1_dataset_IG/valid result_4/randomboxes/hw1_dataset_IG/val2017
    mv result_4/randomboxes/generation_box_image/* result_4/randomboxes/hw1_dataset_IG/train2017
    rm -r result_4/randomboxes/generation_box_image
    ```

* Train the models
  * First, change to the ```DAB-DETR``` directory
    ```cd DAB-DETR```
  
  * Then, train the models and put the result in the ```../result_4```
    1. Original Boxes(with Text Grounding)

      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/originalboxes/results_TG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/originalboxes/hw1_dataset_TG \
      --resume ../checkpoint.pth 
      ```
    
    2. Original Boxes(with Image Grounding)
    
      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/originalboxes/results_IG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/originalboxes/hw1_dataset_IG \
      --resume ../checkpoint.pth 
      ```
    3. Random Boxes(with Text Grounding)

      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/randomboxes/results_TG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/randomboxes/hw1_dataset_TG \
      --resume ../checkpoint.pth 
      ```
    4. Random Boxes(with Image Grounding)

      ```
      python main.py -m dab_deformable_detr \
      --output_dir ../result_4/randomboxes/results_IG \
      --batch_size 1 \
      --epochs 150 \
      --lr_drop 40 \
      --coco_path ../result_4/randomboxes/hw1_dataset_IG \
      --resume ../checkpoint.pth 
      ```
* Evaluation
  * First, output the prediction of validation dataset. We first copy ```4-2.py``` to ```DAB-DETR``` directory, then change the directory to ```DAB-DETR```
    ```
    cp 4-2.py DAB-DETR/
    cd DAB-DETR
    ```
  * Then, output the prediction json file

    1. Original Boxes(with Text Grounding)

      ```
      python 4-2.py \
      ../result_4/originalboxes/results_TG/config.json \
      ../result_4/originalboxes/results_TG/checkpoint.pth \
      ../result_4/originalboxes/hw1_dataset_TG/val2017 \
      ../result_4/originalboxes/hw1_dataset_TG/annotations/instances_val2017.json \
      ../result_4/originalboxes/pred_TG.json
      ```

    2. Original Boxes(with Image Grounding)
      
      ```
      python 4-2.py \
      ../result_4/originalboxes/results_IG/config.json \
      ../result_4/originalboxes/results_IG/checkpoint.pth \
      ../result_4/originalboxes/hw1_dataset_IG/val2017 \
      ../result_4/originalboxes/hw1_dataset_IG/annotations/instances_val2017.json \
      ../result_4/originalboxes/pred_IG.json
      ```

    3. Random Boxes(with Text Grounding)
      
      ```
      python 4-2.py \
      ../result_4/randomboxes/results_TG/config.json \
      ../result_4/randomboxes/results_TG/checkpoint.pth \
      ../result_4/randomboxes/hw1_dataset_TG/val2017 \
      ../result_4/randomboxes/hw1_dataset_TG/annotations/instances_val2017.json \
      ../result_4/randomboxes/pred_TG.json
      ```

    4. Random Boxes(with Image Grounding)
      ```
      python 4-2.py \
      ../result_4/randomboxes/results_IG/config.json \
      ../result_4/randomboxes/results_IG/checkpoint.pth \
      ../result_4/randomboxes/hw1_dataset_IG/val2017 \
      ../result_4/randomboxes/hw1_dataset_IG/annotations/instances_val2017.json \
      ../result_4/randomboxes/pred_IG.json
      ```
  
  * Evaluate the prediction: initially, we have to go back to ```CVPDL_HW3``` directory, and then conduct the evaluation
    ```
    cd ..
    ```

    1. Original Boxes(with Text Grounding)
      
      ```
      python 4-3.py result_4/originalboxes/pred_TG.json \
      result_4/originalboxes/hw1_dataset_TG/annotations/instances_val2017.json
      ```

    2. Original Boxes(with Image Grounding)
      
      ```
      python 4-3.py result_4/originalboxes/pred_IG.json \
      result_4/originalboxes/hw1_dataset_IG/annotations/instances_val2017.json
      ```

    3. Random Boxes(with Text Grounding)
      
      ```
      python 4-3.py result_4/randomboxes/pred_TG.json \
      result_4/randomboxes/hw1_dataset_TG/annotations/instances_val2017.json
      ```

    4. Random Boxes(with Image Grounding)
      
      ```
      python 4-3.py result_4/randomboxes/pred_IG.json \
      result_4/randomboxes/hw1_dataset_IG/annotations/instances_val2017.json
      ```