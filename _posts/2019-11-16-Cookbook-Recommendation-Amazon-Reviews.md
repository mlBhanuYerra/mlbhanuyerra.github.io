---
layout: post
title: Recommendation System for Cookbooks
subtitle: Case Study Using Amazon Book Reviews
image: "/img/cookbooks/CookbookCovers.png"
tags: [Recommendation Systems, Collaborative Recommendation System, Content Based Recommendation System, Natural Language Processing, NLP, Tokenization, Surprise, Topic Modeling, LSA, Latent Semantic Analysis, Matrix Factorization, SVD, Simon Funk]
comments: true
---

### Introduction
Ever since I started watching Great British Baking Show, I added three books to my ever-increasing collection of cookbooks. Do I bake? No sir, I do not! At least not very well. I am not alone in this obsession with cooking shows, and cookbooks. According to NBCNews.com, roughly 18 million cookbooks were sold in 2018 and annual sales of cookbooks have increased by 21% or over in at least the last two years <sup>1, 2, 3</sup>.  The format of these books has changed too – more glossy paper with pictures than a collection of over-running text documenting the recipes.

So, how do we find our cookbooks? We start at Amazon.com reviews, of course!  Combining my fascination with cookbooks and my passion for Data Science, here is my attempt to help my fellow cookbook-enthusiasts in finding their next purchase by building a recommender system based on book collections and user reviews at Amazon.com.  I've used Natural Language Processing (NLP) nmethodologies, and unsupervised learning in clustering books together in finding topics and sub-topics.

Who else would benefit from such a recommender system? Vendors looking to increase their online sales by making appropriate and relevant book suggestions to their customers. And small business book stores that have limited space to store their books, so they can pick titles similar to their previous sales. 


<hr>

### Approach
My approach invloves building a hybrid recommendation system that combines:
- NLP topic modeling techniques to build two content based recommender systems using book titles and book descriptions separately, and
- SVD Matrix-factorization technique (similar to the method Simon Funk popularized for Netflix Prize<sup>4</sup>) for building a collaborative recommender system using user ratings of cookbooks.


#### Dataset
A collection of user reviews and product information from Amazon.com is available from the University of California, San Diego Computer Science department [4].  This dataset spans from 1996 to 2018, and was trimmed to include only cookbook reviews.  This resulted in about 45,000 cookbooks with 29,000 (65%) of them with no user reviews, and the rest - about 16,000 cookbooks - with 428,000 reviews.
Amazon reviews dataset consist of two separate data files in JSON format: 
- Products Table used for cookbook titles and description, and 
- Reviews Table used for extracting user ratings.

An illustation of how data looks on Amazon.com website is shown in Figure 1. 

![Cookbook Data Format on Amazon.com](/img/cookbooks/data_example.png "Amazon.com cookbooks")
*Figure 1: A sample cookbook data point*

#### Model Steps
The data processing and modeling involved a number of steps, each with variations as shown below:
1. Tokenization:
- Lowercase
- Remove non-alphanumeric letters
- Multi-word Expressions (fat-loss, low-carb etc.)
- Remove stopwords
- Lemmatization
2. Vectorization:
- Count vectorization
- <i><b>TF-IDF</b></i> (term frequency - inverse document frequency)
3. Dimensionality Reduction/Topic-Modeling
Reduced vectorized matrices from step 2 to a 10-dimensional space.
- <i><b>Latent Semantic Analysis (LSA)</b></i>
- Non-negative Matrix Factorization
- Latent Dirichilet Allocation

A K-means clustering algorithm was tested after the dimensionality reduction step, but was later determined to be not too beneficial to the cookbook recommendation system. The model that produceds best results (in qualitative terms) is a combination of TF-IDF and LSA.

A number of tools and technologies were used in performing these steps and they include:
- Python
- Sci-kit learn
- Pandas
- Surprise, Python's Recommendation System Engine
- NLTK (Natrual Language ToolKit)

The three stages I've used in building my hybrid recommedation system are as shown below. Note that each stage is distinct and does not share any inputs.  This was intentional to maximize what models can learn from each of those input categories in isolation. An alternative considered was to inform each step what the previous steps have learned but that makes the approach slightly larger than the scope of this project and I hope to explore that at a later date.

![Stages of Recommendation System](/img/cookbooks/Stages.png "Three Pronged Recommendation System")
*Figure 2: Three Stages of the Recommendation System*


#### Additional Data Exploration
Additional exploration of the dimensionality reduction step was performed by projecting the ten dimensonal space onto two dimensions for some select combinations of dimensions to get a better understanding of the topic modeling space.

My intuition was that book titles carry 'quality information' that is distinct from book descriptions, as titles are meant to be succinct, catchy and to the point.  Book tiltes are reviewed, edited and marketed carefully by authors and publishers and carry more weight than cookbook desciptions. On the other hand, book descriptions carry more 'quantity of information' as they use more vocabulary and are more likely to connect diverse concepts. The following two figures confirms this intuitions as the projected space of book titles after LSA is sharp, clean and distinct.  The projected space of book descriptions after LSA on the other hand, spreads and has intersecting branches/blobs (Figure 4). 

![Topic Modeling with Cookbook Titles](/img/cookbooks/Cookbook_Titles_noFigureDesc.png "Topic Modeling with Cookbook Titles")
*Figure 3: Dimensionality Space of Cookbook Titles after LSA: Select projections from 10-dimensional space (Colors are based on K-Means algorithm and are used for illustration purpose only)*

![Topic Modeling with Cookbook Descriptions](/img/cookbooks/Cookbook_Desc_noFigureDesc.png "Topic Modeling with Cookbook Descriptions")
*Figure 4: Dimensionality Space of Cookbook Descriptions after LSA: Select projections from 10-dimensional space (Colors are based on K-Means algorithm and are used for illustration purpose only)*


### Model Results

The model results are presented in a qualitative format for the two content based recommendation systems: Stage 1 & Stage 2. A quantitiave comparison against two 'naive' models were presented for Stage 3 model-based collaborative recommendation system.

For Stage 1 & 2, one approach would have been to introduce a new cookbook and check the 'relevance' of the recommendations.  Instead, I took an explorative approach:
- Select a random book from the train data, 
- Find its closest neighbors determined based on cosine-similarity,
- Display the book covers of those neighbors and visually examine for book categories/topics.

This explorative approach provided a means of discovering what the recommendation systems learned about cookbooks.

The results are quite fascinating to say the least. And I will make an attempt to visually present these in this section. As expected from figures 3 & 4, this explorative approach also revealed that book titles tend to recommend a "narrow" range of options, while book descriptions provide more broader perspective.

Content based recommendation systems often suffer from "information confinement area" problem where the same items are repeatedly suggested to the user based on their previous likes and purchases. This problem leads to a missed opportunity in helping the users explore more items that they might be interested in and perhaps would have lead to a purchase. Using two different recommendation systems for cookbooks will definitely address this issue.

#### Results: Content-based Recommendation System using Book Titles

The model discovered a number of topics and sub-topics of cookbooks. There are nuemerous categories and is impractical to explore all that the model learned!! A few categories that stood out are below:

- Cakes
- Cupcakes
- Desserts with Sugar in their titles
- Sweets
- Vineyards
- Wine & Food
- Beer
- Cocktails
- Seafood
- Pressure Cooker
- Slow Cooker
- Italian
- Souther Food (Includes American southern food and italian southern!)
- Chai & Tea

![Recommendation System with Titles](/img/cookbooks/titles/MetisPrj4_BookTitles.gif "Recommendation System with Titles")
*Figure 5: Content-based Recommendation System using Book Titles*

#### Results: Content-based Recommendation System using Book Descriptions

A few categories that model discovered are:

- Baking
- Dessersts
- Chocolate
- Cheese
- Heralty Cooking
- Holiday Cooking
- Gluten-free, Paleo etc.
- Pies & Tarts
- Vegan

![Recommendation System with Descriptions](/img/cookbooks/desc/MetisPrj4_BookDesc.gif "Recommendation System with Descriptions")
*Figure 6: Content-based Recommendation System using Book Descriptions*


#### Results: Collaborative Recommendation System using Book User Ratings


<hr>
### Conclusions
The primary conclusions from the above image classification analysis are:
1. Protyping a classification model using pretrained CNN features is quite effective and easier than fully building a deep neural network from scratch.
2. Error analysis is quite useful, and provides insights on how models can be employed. 

<br>
<br>
<hr>
### References
1. [Cookbook Sales Are Jumping, Which Is Great News For Shops That Specialize In Them, Forbes, March 2019.](https://www.forbes.com/sites/michelinemaynard/2019/03/10/cookbook-sales-are-jumping-which-is-great-news-for-shops-that-specialize-in-them/#a2e177b6e54d)
2. [Culinary Bookstores Feed Local Appetites, Publishers Weekly, March 2019.](https://www.publishersweekly.com/pw/by-topic/industry-news/bookselling/article/79410-culinary-bookstores-feed-local-appetites.html?fbclid=IwAR2UhWRyM8pwOOgUY1f2YG1aT3u4Jrjs_hEhY8qqSmtWEzUy8au-b5nNPRY)
3. [Recipe for success: Cookbook sales survive shift to digital media, NBCNews.com, August 2018.](https://www.nbcnews.com/business/consumer/recipe-success-cookbook-sales-survive-shift-digital-media-n900621)
4. [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)
5. [Introduction to recommender systems, Overview of some major recommendation algorithms.](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)










