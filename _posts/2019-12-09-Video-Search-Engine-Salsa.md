---
layout: post
title: Building a Video Search Engine
subtitle: A case study using Salsa dance videos
image: "/img/VideoSearch/red_female_2.jpg"
tags: [Video search engine, search enginer, video, conten-based recommendation systems, recommendation system, LSA, Latent Semantic Analysis, Matrix Factorization, OpenPose, Pose Estimation, Computer Vision, Neual Networks, 2D Pose Estimates, Search, video, YouTube, Salsa, Salsa Videos]
comments: true
---

### Introduction
A natural progression in the field of computer vision following unprecedented progress in image classification tasks is towards video and video understanding, especially how it relates to identifying human subjects and activities.  A number of datasets and benchmarks are being estblished in this area<sup>1</sup>.

In parallel, further progess is being made in 2D image related computer vision tasks such as fine-grained classification, image segmention, 3D image construction, robot vision, scene flow estimation and <i><b>human pose estimation</b></i>.

As part of final Data Science project at Metis bootcamp, I've decided to marry these two parallel tracks - video and human pose estimation in specific - to create a content-based video search engine.  Since applying 2D human pose estimation for video search is a novel idea with "no proof of concept", simplification of the problem space is in order!  In addition, a fair amount of experminetation in a "lab" setting is warranted for this approach.

To simplify my video search engine problem, I choise Salsa dance videos on YouTube. Doesn't make sense? Read on to see how the salsa videos I picked from YoutTube channel of "World Salsa Summit" simplifies my video search problem:
1. Single dancers
2. Single camera with fixed location - only angle and zoom level are included in the processed footage.
3. 70 minutes of video footage, formatted into 3 second clips.
4. Reduce the video frame rate to 8 frames per second
5. Keep the video resolution fixed at 640 (width) X 360 (height) pixels.
<hr>

### Video
Users on YouTube, the sedond largest search enginer after Goolge, watch over 1 billion hours of video every single day. Users on Facebook, the most popular social networking site in the world, watch about 100 million hours of video every single day!!  These platforms are interested in providing tools to their users in searching and discovering interesting and relevant material.

The tools these platforms provide for searching primarily use a video's metadata (location, time, content creator etc.), titles, descriptions, transcripts (either user created or machine generated from audio), user ratings, user comments etc. in retrieving 'similar' results. These search tools do not skim the actual content of the video itself.  Video is not skimmable or indexable for searches.

The visual features in the videos are too many and they are computationally expensive to index and too slow to retrieve. As a example, if one were to search for the dance steps of the Salsa dancer in figure 1 from other Salsa videos on YouTube, its nearly impossible.  These steps do not have titles, descriptions or audio to transcribe.

![Video](/img/VideoSearch/Val_106_4nHElVbT3HY_plain.mp4)

*Figure 1: Salsa Dance Video (Credit: World Salsa Summit 2016. Dancer: Valentino Sinatra, Italy)*

The video search enginer I built as part of my Metis Project 5, is based on 70 minutes of processed Salsa dance videos from YouTube and will return a search match as shown in the figure below.  The Salsa steps performed by Yeiufren are similar to Valentino's (on the left) that they both walk back to the center of the stage and turn towards the left of the screen.

![Video](/img/VideoSearch/Val_106_Result.mp4)

*Figure 2: A search results from the Video Search Enginer (Credit: World Salsa Summit 2016. Dancers: Valentino Sinatra, Italy; Yeifren Mata, Venezuela; Adriano Leropoli, Montreal)*


#### Dataset


#### Model Steps
The data processing and modeling involved a number of steps, each with variations as shown below:
1. Vectorization:
- Count vectorization
- <i><b>TF-IDF</b></i> (term frequency - inverse document frequency)
2. Dimensionality Reduction/Topic-Modeling
Reduced vectorized matrices from step 2 to a 10-dimensional space.
- <i><b>Latent Semantic Analysis (LSA)</b></i>
- Non-negative Matrix Factorization
- Latent Dirichilet Allocation


A number of tools and technologies were used in performing these steps and they include:
- Python
- Sci-kit learn
- Pandas
- Surprise, Python's Recommendation System Engine
- NLTK (Natural Language ToolKit)


#### Additional Data Exploration


### Model Results



#### Results: 


<hr>
### References
1. [YouTube-8M: A Large-Scale Video Classification Benchmark](https://arxiv.org/pdf/1609.08675.pdf)
2. [Culinary Bookstores Feed Local Appetites, Publishers Weekly, March 2019.](https://www.publishersweekly.com/pw/by-topic/industry-news/bookselling/article/79410-culinary-bookstores-feed-local-appetites.html?fbclid=IwAR2UhWRyM8pwOOgUY1f2YG1aT3u4Jrjs_hEhY8qqSmtWEzUy8au-b5nNPRY)
3. [Recipe for success: Cookbook sales survive shift to digital media, NBCNews.com, August 2018.](https://www.nbcnews.com/business/consumer/recipe-success-cookbook-sales-survive-shift-digital-media-n900621)
4. [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)
5. [Introduction to recommender systems, Overview of some major recommendation algorithms.](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)










