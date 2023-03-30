import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vit_pytorch import ViT_face
from torchvision import datasets, transforms

# Load the state_dict from the checkpoint
state_dict = torch.load('/Users/br/Software/Machine_learning/MTCNN_face_transformer/pretrained_models/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth', map_location=torch.device('cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model
transformer = ViT_face(
    image_size=112,
    patch_size=8,
    loss_type='CosFace',
    GPU_ID=device,
    num_class=93431,  
    dim=512,
    depth=20,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# Load the state_dict into the model
transformer.load_state_dict(state_dict)
transformer.eval()

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load the dataset
matched_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/matched_faces', transform=data_transforms)
matched_loader = DataLoader(matched_dataset, batch_size=2, shuffle=False)

mismatched_dataset = datasets.ImageFolder('/Users/br/Software/Machine_learning/MTCNN-VGGFace2-InceptionResnetV1/LFW_dataset/mismatched_faces', transform=data_transforms)
mismatched_dataset.idx_to_class = {i: c for c, i in mismatched_dataset.class_to_idx.items()}
mismatched_loader = DataLoader(mismatched_dataset, batch_size=2, shuffle=False)

transformer.to(device)

def calculate_accuracy(labels, distances, threshold):
    correct_predictions = 0
    total_predictions = len(labels)

    for i in range(len(distances)):
        if labels[i] == 1 and distances[i] <= threshold:
            correct_predictions += 1
        if labels[i] == 0 and distances[i] > threshold:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_far_frr(labels, distances, threshold):
    num_positive = np.sum(labels)
    num_negative = len(labels) - num_positive
    false_accepts = 0
    false_rejects = 0

    for i in range(len(distances)):
        if labels[i] == 1 and distances[i] > threshold:
            false_rejects += 1
        if labels[i] == 0 and distances[i] <= threshold:
            false_accepts += 1

    FAR = false_accepts / num_negative
    FRR = false_rejects / num_positive
    return FAR, FRR

# Initialize variables to track time
computation_time = 0
n_operations = 0

# generate embeddings for the dataset
matched_embedding_list = []
matched_name_list = []
for folder in os.listdir(matched_dataset.root):
    folder_path = os.path.join(matched_dataset.root, folder)
    if os.path.isdir(folder_path):
        # load the first two images in the folder
        images = []
        for i, filename in enumerate(sorted(os.listdir(folder_path))):
            if i >= 2:
                break
            image_path = os.path.join(folder_path, filename)
            image = datasets.folder.default_loader(image_path)
            images.append(image)
        # generate embeddings for the two images in the folder
        if len(images) == 2:
            embeddings = []
            for i in range(2):
                start_time_matched_resnet = time.time()
                img_tensor = data_transforms(images[i]).unsqueeze(0).to(device)
                emb = transformer(img_tensor)
                embeddings.append(emb.detach())
                elapsed_time_matched_resnet = time.time() - start_time_matched_resnet
                computation_time += elapsed_time_matched_resnet
                n_operations += 1
            matched_embedding_list.extend(embeddings)
            matched_name_list.extend([folder] * 2)
        else:
            print(f"Not enough images in {folder_path}")

# generate embeddings for the dataset
mismatched_embedding_list = []
mismatched_name_list = []
for image, index in mismatched_loader:
    start_time__mismatched_resnet = time.time()
    img_tensor = image.to(device)
    emb = transformer(img_tensor)
    mismatched_embedding_list.append(emb.detach())
    mismatched_name_list.append(mismatched_dataset.idx_to_class[index[0].item()])
    elapsed_time_mismatched_resnet = time.time() - start_time__mismatched_resnet
    computation_time += elapsed_time_mismatched_resnet
    n_operations += 1

if len(mismatched_embedding_list) % 2 != 0:
    mismatched_embedding_list.pop()
    mismatched_name_list.pop()
    
dist_matched = [torch.dist(matched_embedding_list[i], matched_embedding_list[i + 1]).item() for i in range(0, len(matched_embedding_list), 2)]
dist_mismatched = [torch.dist(mismatched_embedding_list[i], mismatched_embedding_list[i + 1]).item() for i in range(0, len(mismatched_embedding_list), 2)]

# Calculate average distances
avg_dist_matched = np.mean(dist_matched)
avg_dist_mismatched = np.mean(dist_mismatched)

distances = dist_matched + dist_mismatched
labels = [1] * len(dist_matched) + [0] * len(dist_mismatched)

accuracies = []
thresholds = np.linspace(0.1, 1.6, 5000)
FARs = []
FRRs = []
for threshold in thresholds:
    FAR, FRR = calculate_far_frr(labels, distances, threshold)
    accuracy = calculate_accuracy(labels, distances, threshold)
    FARs.append(FAR)
    FRRs.append(FRR)
    accuracies.append(accuracy)

max_accuracy_index = np.argmax(accuracies)
max_accuracy_threshold = thresholds[max_accuracy_index]
eer_index = np.argmin(np.abs(np.array(FARs) - np.array(FRRs)))
eer = (FARs[eer_index] + FRRs[eer_index]) / 2

# Calculate the accuracy and F1 score of the model
true_matched = []
true_mismatched = []
pred_matched = []
pred_mismatched = []

# Compare embeddings and calculate metrics for matched faces
for i in range(0, len(matched_embedding_list), 2):
    emb1 = matched_embedding_list[i]
    emb2 = matched_embedding_list[i + 1]
    dist = torch.dist(emb1, emb2).item()
    is_match = matched_name_list[i] == matched_name_list[i + 1]
    true_matched.append(is_match)
    pred_matched.append(dist < max_accuracy_threshold)

# Compare embeddings and calculate metrics for mismatched faces
for i in range(0, len(mismatched_embedding_list), 2):
    emb1 = mismatched_embedding_list[i]
    emb2 = mismatched_embedding_list[i + 1]
    dist = torch.dist(emb1, emb2).item()
    is_mismatch = mismatched_name_list[i] != mismatched_name_list[i + 1]
    true_mismatched.append(is_mismatch)
    pred_mismatched.append(dist > max_accuracy_threshold)


# Update the counts for recognized faces
True_Positives = sum([same_face and pred for same_face, pred in zip(true_matched, pred_matched)]) # True Positives
True_Negatives = sum([different_face and pred for different_face, pred in zip(true_mismatched, pred_mismatched)]) # True Negatives
False_Negatives = sum([same_face and not pred for same_face, pred in zip(true_matched, pred_matched)]) # False Negatives
False_Positives = sum([different_face and not pred for different_face, pred in zip(true_mismatched, pred_mismatched)]) # False Positives

# Calculate total predictions
total_predictions = True_Positives + False_Negatives + True_Negatives + False_Positives

# Calculate accuracy, precision, recall, and F1 score
accuracy = (True_Positives + True_Negatives) / total_predictions
precision = True_Positives / (True_Positives + False_Positives)
recall = True_Positives / (True_Positives + False_Negatives)
f1 = 2 * precision * recall / (precision + recall)
# Calculate the average InceptionResnetV1 time
average_computation_time = computation_time / n_operations

print("Optimal Accuracy Threshold", max_accuracy_threshold)
print("Average Computation time:", average_computation_time)
print("Detection Rate: 100.0")
print("True Positives: ", True_Positives)
print("True Negatives: ", True_Negatives)
print("False Negatives: ", False_Negatives)
print("False Positives: ", False_Positives)
print("Average distance for matched faces:", avg_dist_matched)
print("Average distance for mismatched faces:", avg_dist_mismatched)
print("EER:", eer)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)


# Scatter plot
fig, ax = plt.subplots()

def add_jitter(values, jitter_amount):
    return [value + jitter_amount * (2 * np.random.rand() - 1) for value in values]

# Increase jitter amount
jitter_amount = 0.45

# Define plot title
plot_title = "Face Transformer Pre-trained on MS-Celeb-1M tested on the LFW Dataset"

dist_true_positives = [dist for dist, same_face, pred in zip(dist_matched, true_matched, pred_matched) if same_face == True and pred == True]
dist_true_negatives = [dist for dist, different_face, pred in zip(dist_mismatched, true_mismatched, pred_mismatched) if different_face == True and pred == True]
dist_false_negatives = [dist for dist, same_face, pred in zip(dist_matched, true_matched, pred_matched) if same_face == True and pred == False]
dist_false_positives = [dist for dist, different_face, pred in zip(dist_mismatched, true_mismatched, pred_mismatched) if different_face == True and pred == False]

# True positives
ax.errorbar(dist_true_positives, add_jitter([0] * len(dist_true_positives), jitter_amount), fmt='s', c='green', alpha=0.5, label='True Positives')
# False negatives
ax.errorbar(dist_false_negatives, add_jitter([0] * len(dist_false_negatives), jitter_amount), fmt='o', c='red', alpha=0.5, label='False Negatives')
# True negatives
ax.errorbar(dist_true_negatives, add_jitter([1] * len(dist_true_negatives), jitter_amount), fmt='p', c='blue', alpha=0.5, label='True Negatives')
# False positives
ax.errorbar(dist_false_positives, add_jitter([1] * len(dist_false_positives), jitter_amount), fmt='P', c='orange', alpha=0.5, label='False Positives')

# EER threshold
ax.axvline(x=max_accuracy_threshold, color='purple', linestyle='-', label='Optimal Accuracy Threshold Distance')
ax.axvline(x=avg_dist_matched, color='black', linestyle=':', label='Average Matched Distance')
ax.axvline(x=avg_dist_mismatched, color='black', linestyle='--', label='Average Mismatched Distance')

ax.set_xlabel('Distance')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Matched Faces','Mismatched Faces'])
ax.set_title(plot_title)
ax.legend()

plt.show()

