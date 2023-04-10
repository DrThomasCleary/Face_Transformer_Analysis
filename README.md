# Face_Transformer_Analysis

Please refer to my other repository for details on how to set your data for face verification in order to test your model performance on matched face pairs 
and mismatched face pairs: https://github.com/DrThomasCleary/MTCNN_InceptionResnetV1_VGGFace2

My results for the face Transformer:
![Screenshot 2023-04-10 at 01 10 53](https://user-images.githubusercontent.com/118690399/230803032-0185c4d2-7f2d-40ab-a67e-0a4dfead673f.png)

A True Positive, True Negative, False Positive and False Negative plot:
![Screenshot 2023-04-01 at 00.11.35.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1843172e-8885-457a-83b5-13a2206cc1dd/Screenshot_2023-04-01_at_00.11.35.png)



These are decent results for an architecture so young but the paper where the face transformer was introduced noted "We demonstrate that Face Transformer models trained on a large-scale database, 
MS-Celeb-1M, achieve comparable performance as CNN with similar number of parameters and MACs." Though this may be true but i found a large flaw in their model when testing on Blurred images such as:

![Screenshot 2023-04-10 at 01 15 42](https://user-images.githubusercontent.com/118690399/230803260-2a9585e9-50f5-4b3f-aae7-c103139146f0.png)

The results:
![Screenshot 2023-04-10 at 01 16 20](https://user-images.githubusercontent.com/118690399/230803292-94df40ac-7499-4747-bbc2-697019c8b43e.png)


Which may seem reasonable if looking at only the Accuracy, or f1 score, but once plotting the graphs, it became apparent what is wrong:
![Screenshot 2023-04-04 at 01.13.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55266f62-190d-43c5-ae39-c49d305bf93d/Screenshot_2023-04-04_at_01.13.44.png)

The Recall increases as the Blurry intensity inreases and precision decreases. Looking closer at another graph, it becomes clear what is happening:
![Screenshot 2023-04-01 at 01.54.59.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45be1fb4-23c0-4737-87e8-d7a7b56b5787/Screenshot_2023-04-01_at_01.54.59.png)

The model shifted the entire confidence scores of all tests closer to 0! This would give more True positive results and decrease false negatives (hence a higher recall) 
but also greatly increase the amount of False positives (hence a lower precision). 
A similar phonomenom is observed with decrease in resolution which makes it clear that this model is poor at generalising and need further work to increase robustness. I suggest:
-Augment the training data with more low-resolution and blurry images
-Fine-tune the model on low-resolution or blurry images
-Apply adaptive computation time strategies
-Modify the loss function to consider the distance between matched and mismatched pairs more explicitly, or explore other loss functions such as ArcFace or SphereFace
-Experiment with different model architectures or hyperparameter configurations to improve the model's robustness
-Incorporate attention mechanisms to focus on the most discriminative facial features, especially in challenging conditions
