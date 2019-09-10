# Modeling Film Similarity with NLP and Clustering Algorithms

![Films](https://dz7u9q3vpd4eo.cloudfront.net/wp-content/legacy/posts/4741addc-1756-41f1-b2ff-c3ea98d77647.jpg)

This project entails using Natural Language Processing (NLP) techniques (ie: regular expressions, tokenization, stemming, vectorization for TF-IDF) and clustering algorithms (KMeans and Hierarchical clustering) to model the "similarities" between films based on their plots provided by IMDb and Wikipedia. The dataset contains the titles of the top 100 movies on IMDb. Steps taken include the following:

## 1. Combining Plot Data
Merging the Wikipedia and IMDb plots

## 2. Tokenization
Breaking out the sentences and words in the plots

## 3. Stemming
Stemming the word tokens to their base form

## 4. Creating TF-IDF
Creating the term frequency inverse document frequency object

## 5. Computing Distance and Clusters
Computing the euclidean distance of common terminology across plots + leveraging the KMeans algorithm for clustering the distances

## 6. Calculating Similarity
Using max / complete linkage to compute the similarity between film plots

## 7. Visualization
Creating a dendrogram to visualize the films clustered together and their respective hierarchies. 

