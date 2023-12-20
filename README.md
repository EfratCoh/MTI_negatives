
MicroRNAs (miRNAs) are small non-coding RNAs that regulate gene expression post-transcriptionally via base-pairing with complementary sequences on messenger RNAs (mRNAs). Computational approaches that predict miRNA target interactions (MTIs) facilitate the process of 
narrowing down potential targets for experimental validation. The availability of new datasets of high-throughput, direct MTIs has led to the development of machine learning (ML) based methods for MTI prediction. To train an ML algorithm, there is a need to 
supply entries from all class labels (i.e., positive and negative). 
Currently, no high-throughput assays exist for capturing negative examples, hindering effective classifier construction. Therefore, current ML approaches must rely on artificially generated negative examples for training. 
Moreover, the lack of uniform standards for generating such data leads to biased results and hampers comparisons between studies.
In this comprehensive study, we investigated the impact of different methods to generate negative data on the classification of true MTIs in animals. 
Our study relies on training ML models on a fixed positive dataset in combination with different negative datasets and evaluating their intra- and cross-dataset performance. 
As a result, we were able to examine each method independently and evaluate ML models’ sensitivity to the methodologies utilized in negative data generation. To achieve a deep understanding of the performance results, 
we analyzed unique features that distinguish
between datasets. In addition, we examined whether unsupervised one-class classification models that utilize solely positive interactions for training are suitable for the task of MTIs classification. 
We demonstrate the importance of negative data in MTI classification, analyze specific methodological characteristics that differentiate negative datasets, and highlight the challenge of ML models generalizing interaction rules 
from training to testing sets derived from different approaches.  This study provides valuable insights into the computational prediction of MTIs that can be further used to establish standards in the field.

The code is 
