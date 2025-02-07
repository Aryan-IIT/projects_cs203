## Active Learning 

Supervised learning relies on a fully labeled dataset, where each input is paired with a correct output, allowing models to learn clear patterns. Semi-supervised learning, in contrast, uses a mix of labeled and unlabeled data—typically a small set of labeled examples and a larger pool of unlabeled ones—leveraging unsupervised techniques to improve learning efficiency. Active learning takes this further by allowing the model to selectively query an oracle (e.g., a human annotator) for labels on the **most informative or uncertain samples**, reducing the labeling effort while maximizing learning performance.

### Two types of active learning 

- Stream Based active learning => Unlabeled example by example, query its label or ignore it
- Pool based active learning => Given: a large unlabeled pool of
examples, Rank examples in order of informativeness, Query the labels for the most informative example(s)

### Query Selection Strategies 

### Uncertainty-Based Active Learning Strategies  

In uncertainty-based active learning, the model selects samples it is most uncertain about for labeling. Below are three common strategies:  

1. **Least Confident**: Selects the sample where the model’s highest predicted probability is the lowest, indicating maximum uncertainty.  
   \[
   x^* = \arg\min_x P_{\theta}(\hat{y} | x)
   \]  

2. **Smallest Margin**: Chooses the sample where the difference between the top two predicted probabilities is smallest, meaning the model is nearly undecided between them.  
   \[
   x^* = \arg\min_x \left( P_{\theta}(y_1 | x) - P_{\theta}(y_2 | x) \right)
   \]  

3. **Entropy**: Picks the sample with the highest entropy in the predicted distribution, representing maximum disorder in the model’s confidence.  
   \[
   x^* = \arg\max_x -\sum_{i} P_{\theta}(y_i | x) \log P_{\theta}(y_i | x)
   \]  

Each method captures uncertainty differently, with entropy providing a holistic measure across all classes.

### Query by Committee (QBC) in Active Learning  

Query by Committee (QBC) is an active learning strategy where multiple models (a "committee") are trained on the same labeled data but with variations (e.g., different initializations, subsets, or architectures). The model selects samples that induce the highest disagreement among the committee members for labeling. Common disagreement measures include:  

1. **Vote Entropy**: Measures disagreement based on the distribution of predicted labels across committee members. Higher entropy indicates greater uncertainty.  
   \[
   x^* = \arg\max_x -\sum_{i} \frac{V(y_i)}{C} \log \frac{V(y_i)}{C}
   \]  
   where \( V(y_i) \) is the number of votes for class \( y_i \), and \( C \) is the total number of committee members.  


QBC efficiently reduces labeling efforts by selecting samples that challenge the diversity of model opinions, improving learning efficiency.
KL Divergence and Consensus margin are also QBC ways. 

### Some questions 
	1.	Reusing an Actively Labeled Dataset: Yes, but the dataset may be biased toward uncertain samples selected by the active learning strategy. If the new model differs significantly in architecture or assumptions, it might not generalize well without additional randomly sampled data.
	2.	Consequences of Biased Sampling: The actively labeled dataset may not represent the true data distribution, leading to poor generalization and overfitting to uncertain cases.
    Mitigation Strategies:
	•	Hybrid Sampling: Mix actively selected and randomly sampled data.
	•	Reweighting: Adjust sample weights to match the original distribution.
	•	Transfer Learning: Pretrain on a diverse dataset before fine-tuning on actively selected samples.

## Data Annotation 

