import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve, auc , average_precision_score
from sklearn.preprocessing import label_binarize

class _ECE_criterion(nn.Module ):

    def __init__(self, n_bins=10):

            super(_ECE_criterion, self).__init__()
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
            

    def forward(self, logits, labels):
            # Get objectness scores (index 4) and class logits (indices 5-14)
            
            class_logits = logits[..., :]

            
            class_scores = torch.nn.functional.sigmoid(class_logits)
            confidences = class_scores  # shape (7229 , 10)

            pred_confidence  , predictions = torch.max(confidences , dim = -1) # shape (7229 , 1) 
            
            print('confidece is : ' , pred_confidence[:25])

            print(predictions.shape)

            accuracies = predictions.eq(labels)
            print('accuracy shape is : ',accuracies.shape)
            print("confidence shape is : ",pred_confidence.shape)
            # Calculate bin accuracies and confidences
            

            bin_confidences = []
            bin_accuracies = []
            bin_counts = []

            ece = torch.zeros(1 , device = logits.device)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin =  pred_confidence.gt(bin_lower.item()) * pred_confidence.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = pred_confidence[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                    # for plotting the reliability diagram
                    bin_confidences.append(avg_confidence_in_bin.item())
                    bin_accuracies.append(accuracy_in_bin.item())
                    bin_counts.append(in_bin.sum())

            self.bin_confidences = bin_confidences
            self.bin_accuracies = bin_accuracies
            self.bin_counts = bin_counts
            return ece
    
    def plot_reliability_diagram(self):

        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(self.bin_confidences,self.bin_accuracies, 'b-o', label='Model Calibration')
        for i, count in enumerate(self.bin_counts):
            plt.text(self.bin_confidences[i], self.bin_accuracies[i], f'({self.bin_counts[i]})', fontsize=8)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)
        plt.show()
          






class BrierScore(nn.Module):
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, logits, labels):
        # Convert logits to probabilities using softmax
        probabilities = F.sigmoid(logits)  # Shape: (N, K)
        
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=logits.shape[-1]).float()  # Shape: (N, K)
        
        # Compute squared errors between probabilities and one-hot labels
        squared_errors = (probabilities - one_hot_labels) ** 2  # Shape: (N, K)
        
        # Sum over classes and average over samples
        brier_score = squared_errors.sum(dim=-1).mean()  # Shape: scalar
        
        self.plot_class_brier_scores(logits, labels)
        return brier_score

    def plot_class_brier_scores(self, logits, labels):
        # Define the class names as specified
        class_names = ['Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 
                       'Organic', 'Wood', 'e-Waste', 'Rubble', 'Fabric']
        
        # Ensure the number of classes matches the provided names
        num_classes = logits.shape[-1]
        assert len(class_names) == num_classes, "Number of class names must match the number of classes."

        # Convert logits to probabilities using softmax
        probabilities = F.sigmoid(logits)  # Shape: (N, K)

        # Initialize list to store Brier scores for each class
        class_brier_scores = []

        for class_idx in range(num_classes):
            # Binary labels for this class: 1 if true label is this class, else 0
            binary_labels = (labels == class_idx).float()  # Shape: (N,)

            # Predicted probabilities for this class
            class_probs = probabilities[:, class_idx]  # Shape: (N,)

            # Compute Brier score for this class
            squared_errors = (class_probs - binary_labels) ** 2  # Shape: (N,)
            brier_score = squared_errors.mean()  # Scalar

            class_brier_scores.append(brier_score.item())

        # Plot the Brier scores
        plt.figure(figsize=(12, 6))
        plt.bar(class_names, class_brier_scores, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Brier Score')
        plt.title('Brier Score per Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()







#SoftBinned ECE Loss
class SoftBinnedECELoss(nn.Module):

    def __init__(self, n_bins=15 , init_temp = 1 , inside_temp =1 ):

        super().__init__()

        self.n_bins = n_bins
        self.temp = nn.Parameter(torch.ones(1) * init_temp)
        self.inside_temp = nn.Parameter(torch.ones(1) * inside_temp)
        self.register_buffer('bin_centers' , torch.linspace(0 , 1 , steps  = n_bins ))



    def forward(self , logits , labels):

        probs = F.sigmoid(logits)  # logit shape ( N , class ) 
        conf , preds  = torch.max(probs , dim = 1)

        accs = (preds == labels).float()   #N


        T = self.inside_temp  

        C = conf.unsqueeze(1)
        centers = self.bin_centers.unsqueeze(0)

        g = -((C - centers) ** 2) / T  # N , M   N: confidece sample , M: number of Bins

        u = F.softmax(g  , dim = 1) # N * M

        weights = u.sum(dim = 0) + 1e-8   # M

        wc = (u * C).sum(dim = 0)         # M

        wa = (u * accs.unsqueeze(1)).sum(dim = 0 ) # M

        avg_conf = wc / weights
        avg_acc = wa / weights

        ece = (weights * torch.abs(avg_conf - avg_acc)).sum()

        return ece


class ClassificationMetrics:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_precision_recall(self, logits, labels):
        "compute micro average precision and recall"

        probs = torch.sigmoid(logits)
        probs = probs.cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        labels = labels.cpu().numpy()

        precision = precision_score(labels, predictions, average='micro')
        recall = recall_score(labels, predictions, average='micro')

        return precision, recall

    def compute_per_class_precision_recall(self, logits, labels):
        """Compute per-class precision and recall."""
        probs = torch.sigmoid(logits).cpu().numpy()
        predicted = np.argmax(probs, axis=1)
        labels = labels.cpu().numpy()
        precision = precision_score(labels, predicted, average=None, labels=range(self.num_classes), zero_division=0)
        recall = recall_score(labels, predicted, average=None, labels=range(self.num_classes), zero_division=0)
        return precision, recall

    def compute_roc_curve(self, logits, labels):
        "compute micro average ROC curve and AUC"

        probs = torch.sigmoid(logits)
        probs = probs.cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        labels = labels.cpu().numpy()

        labels_bin = label_binarize(labels, classes=np.arange(self.num_classes))

        fpr, tpr, _ = roc_curve(labels_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    
    def compute_ap_per_class(self , logits , labels):

        # compute average precision per classq

        class_names = ['Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 
                       'Organic', 'Wood', 'e-Waste', 'Rubble', 'Fabric']

        probs = torch.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()

        ap_per_class= {}

        for class_idx in range(self.num_classes):
            class_labels = (labels == class_idx).astype(int)
            ap = average_precision_score(class_labels, probs[:, class_idx])
            ap_per_class[class_names[class_idx]] = ap


        return ap_per_class


    def compute_map(self , ap_per_class):
        """Compute mean Average Precision (mAP) across all classes."""
        
        map_value = np.mean(list(ap_per_class.values()))
        return map_value