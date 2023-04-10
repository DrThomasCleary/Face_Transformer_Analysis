# Face_Transformer_Analysis

Please refer to my other repository for details on how to set your data for face verification in order to test your model performance on matched face pairs 
and mismatched face pairs: https://github.com/DrThomasCleary/MTCNN_InceptionResnetV1_VGGFace2

My results for the face Transformer:

![Screenshot 2023-04-10 at 01 10 53](https://user-images.githubusercontent.com/118690399/230803032-0185c4d2-7f2d-40ab-a67e-0a4dfead673f.png)

A True Positive, True Negative, False Positive and False Negative plot:
![Screenshot 2023-04-10 at 01 42 28](https://user-images.githubusercontent.com/118690399/230804658-b0600838-de0f-44f3-b483-ac0d7d7186a7.png)


The results are respectable for such a new architecture, but the paper introducing the face transformer stated, "We demonstrate that Face Transformer models trained on a large-scale database, MS-Celeb-1M, achieve comparable performance as CNN with a similar number of parameters and MACs." While this claim may hold some truth, I discovered a significant flaw in their model when testing it on blurred images, as follows:

![Screenshot 2023-04-10 at 01 15 42](https://user-images.githubusercontent.com/118690399/230803260-2a9585e9-50f5-4b3f-aae7-c103139146f0.png)

The results:

![Screenshot 2023-04-10 at 01 16 20](https://user-images.githubusercontent.com/118690399/230803292-94df40ac-7499-4747-bbc2-697019c8b43e.png)


At first glance, focusing solely on accuracy or F1 scores might make the model seem reasonable. However, upon plotting the graphs, the underlying issue becomes evident:
![Screenshot 2023-04-10 at 01 41 35](https://user-images.githubusercontent.com/118690399/230804603-8dc1e28d-d05e-4cc9-ac6a-b3b0aa233354.png)

As the blurriness intensity increases, the Recall rises while the Precision decreases. By examining another graph more closely, the underlying issue becomes apparent:
![Screenshot 2023-04-10 at 01 42 00](https://user-images.githubusercontent.com/118690399/230804625-b7a4ce1a-48ff-4649-81ac-a867ecd95c51.png)

The model shifts the confidence scores of all tests closer to 0! This results in more True Positive outcomes and a decrease in False Negatives (leading to higher recall), but it also significantly increases the number of False Positives (resulting in lower precision). A similar phenomenon is observed with a decrease in resolution, which makes it evident that this model struggles with generalization and requires further work to enhance its robustness. I suggest the following:

-Augment the training data with more low-resolution and blurry images.

-Fine-tune the model on low-resolution or blurry images.

-Apply adaptive computation time strategies.

-Modify the loss function to take into account the distance between matched and mismatched pairs more explicitly, or explore other loss functions such as 
ArcFace or SphereFace.

-Experiment with different model architectures or hyperparameter configurations to improve the model's robustness.

-Incorporate attention mechanisms to focus on the most discriminative facial features, especially in challenging conditions.


I would highly recommend someone to retrain the transformer with those faults in mind for better results and advancement in using the transformer architecture for face recognition. 
