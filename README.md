# TPHCA Dataset Introduction

## Dataset

### Detailed statistics of benchmark dataset

| Dataset     | #Tar    | #Sent  | #Neg   | #Neu   | #Pos   | #Len    |
|-------------|---------|---------|---------|---------|---------|----------|
| Laptop      | 2,720   | 1,792   | 914     | 579     | 1,227   | 18.60    |
| Restaurants | 4,362   | 2,470   | 921     | 763     | 2,678   | 17.26    |
| Twitter     | 6,315   | 6,312   | 1,584   | 3,154   | 1,577   | 20.61    |

**Field Description**
- #Tar: Number of target aspect terms
- #Sent: Number of sentences
- #Neg: Number of negative polarity samples
- #Neu: Number of neutral polarity samples
- #Pos: Number of positive polarity samples
- #Len: Average sentence length

### Dataset Introduction

**Laptop**  
Constructed by Pontiki et al., sourced from laptop-related reviews on English e-commerce platforms. It focuses on aspect-level sentiment annotation for laptop products, covering multiple specific evaluation dimensions such as battery life, performance, and appearance.

**Restaurants**  
Also constructed by Pontiki et al., derived from restaurant-related review texts on consumer review platforms. It focuses on aspect-level sentiment analysis in the catering scenario, with annotated aspect terms including core evaluation dimensions such as food taste, service quality, ambient atmosphere, and price.

**Twitter**  
Constructed by Dong et al. based on corpus from the Twitter social platform. The dataset features scattered aspect terms, colloquial and casual sentiment expressions, and numerous short-text samples, which align with sentiment expression characteristics in real social scenarios.


You can get the original datasets from the following links:

- Laptop: [https://aclanthology.org/S16-1002/](https://aclanthology.org/S16-1002/)
- Restaurants: [https://aclanthology.org/S16-1002/](https://aclanthology.org/S16-1002/)
- Twitter: [https://aclanthology.org/P14-2009/](https://aclanthology.org/P14-2009/)
<br>

## Running

Run the following command for training:<br>
`python main.py`

- Based on the experimental results on roberta-base, TPHCA achieved excellent F1 sources of 88.64, 87.53, and 88.77 on the Laptop, Restaurants, and Twitter datasets, respectively, all of which were better than the existing baseline models.
- Download Pytorch RoBERTa model from Huggingface https://huggingface.co/roberta-base and put in the folder `roberta-base`.
