# 3DAttnVAE-GAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project focuses on the problem of reconstructing 3D shapes from a single 2D image, utilizing a VAE-GAN model. The model is trained on the ShapeNetCore dataset, which consists of over 48,600 3D models across 55 common categories.

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Related Works](#related-works)
- [Methodology](#methodology)
  - [Attention Block](#attention-block)
  - [Variational Encoder](#variational-encoder)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
- [Experiments](#experiments)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Abstract
This project delves into the problem of reconstructing 3D shapes from a single 2D image, leveraging the extensive ShapeNetCore dataset comprising over 48,600 3D models across 55 common categories. The primary objective is to assess the performance of 3D reconstruction algorithms using a single image and compare their effectiveness against existing approaches. The project encompasses crucial stages, including data setup and preprocessing, the development of a robust methodology for 3D shape reconstruction, conducting comprehensive experiments, analyzing and presenting the obtained results, and outlining a roadmap for future research. By examining the challenges associated with capturing 3D shape information from a single image, this work contributes to advancing the field of 3D shape reconstruction. The evaluation of diverse reconstruction algorithms using the ShapeNetCore dataset allows for a rigorous assessment of their performance and offers insights into the state-of-the-art techniques.

## Introduction
In recent years, the availability of 3D data has expanded exponentially, leading to advancements in algorithms that enhance our understanding of the three-dimensional world. However, numerous challenges persist in dealing with 3D representations and processing, leaving several research issues open for exploration. Among these challenges, 3D shape reconstruction from a single image stands as a pivotal problem, holding great potential for applications in virtual and augmented reality, robotics, and autonomous vehicles.

The pursuit of capturing 3D shapes from 2D images has been a decades-old problem. While 3D sensors exist, they are often costly and not universally accessible. As a result, the quest to develop techniques capable of reconstructing 3D shapes from single images, without relying on additional sensors or multiple views, remains a compelling research direction. This raises the fundamental question: Can we accurately reconstruct 3D shape information from a single image while minimizing assumptions?

In this project, our main objective is to evaluate the performance of 3D reconstruction algorithms on the ShapeNet dataset (Chang et al., 2015) - a widely used benchmark for 3D shape analysis and understanding. To conduct our experiments, we utilized the ShapeNetCore subset, encompassing approximately 48,600 3D models spanning 55 common categories. For training, validation, and testing purposes, the dataset is already split into 70%, 10%, and 20% respectively. As our output 3D representations, we chose voxels - a popular choice in 3D reconstruction tasks.

By delving into the complexities of 3D shape reconstruction from a single image, we aim to know the advancement of this field and explore potential solutions to the challenges posed.

## Related Works
Recent advancements in 3D shape generation have led to the development of novel frameworks such as 3D Generative Adversarial Network (3D-GAN) [9]. 3D-GAN generates high-quality 3D objects from a probabilistic space by leveraging volumetric convolutional networks and generative adversarial nets. The adversarial criterion used in the framework enables the generator to capture object structures implicitly and synthesize high-quality 3D objects. Moreover, the generator establishes a mapping from a low-dimensional probabilistic space to the space of 3D objects, allowing us to sample objects without a reference image or CAD models. The discriminator of 3D-GAN also provides a powerful 3D shape descriptor, learned without supervision, which has wide applications in 3D object recognition.

Pix3D [5] is another related work, which is a large-scale benchmark dataset of diverse image-shape pairs with pixel-level 2D-3D alignment. Unlike existing datasets, Pix3D contains real-world images and 3D models with precise alignment. The authors of this paper claim to have calibrated the evaluation criteria for 3D shape reconstruction through behavioral studies and used them to systematically benchmark state-of-the-art reconstruction algorithms on Pix3D. Additionally, the authors designed a novel model that performs 3D reconstruction and pose estimation simultaneously, achieving state-of-the-art performance on both tasks.

AtlasNet [4] introduces a method for learning to generate the surface of 3D shapes. The approach represents a 3D shape as a collection of parametric surface elements and naturally infers a surface representation of the shape. AtlasNet offers significant advantages, such as improved precision and generalization capabilities, and the ability to generate shapes of arbitrary resolution without memory issues. The framework demonstrates its effectiveness and outperforms strong baselines on the ShapeNet benchmark for applications including auto-encoding shapes and single-view reconstruction from still images. Furthermore, AtlasNet shows potential in various other applications, such as morphing, parametrization, super-resolution, matching, and co-segmentation.

MarrNet [8] addresses the challenge of 3D object reconstruction from a single image by proposing an end-to-end trainable model that sequentially estimates 2.5D sketches and 3D object shape. The disentangled, two-step formulation of MarrNet offers advantages such as improved transferability from synthetic to real data by recovering 2.5D sketches, and the ability to learn from synthetic data for 3D reconstruction. The framework employs differentiable projective functions to enable end-to-end training on real images without human annotations, achieving state-of-the-art performance in 3D shape reconstruction.

These related works, including 3D-GAN, Pix3D, AtlasNet, and MarrNet, contribute to the field of 3D shape reconstruction from a single image by exploring different approaches, addressing challenges, and achieving notable advancements. The evaluation and comparison of diverse reconstruction algorithms using datasets like ShapeNetCore and Pix3D allow for a rigorous assessment of their performance and provide insights into state-of-the-art techniques. These works collectively contribute to advancing the field and provide valuable knowledge in the area of 3D shape reconstruction.


## C. Methodology

My proposed method for 3D shape reconstruction from a single image is a modified version of VAE GAN that incorporates attention blocks [7]. The combination of CNN-based architectures and Transformer-based architectures has been a topic of interest in the field of machine learning. While CNNs excel at learning spatial features, Transformers are known for their ability to capture sequential patterns, making them effective in language tasks such as natural language processing. Leveraging the strengths of both architectures, my model aims to incorporate attention mechanisms into the 3D shape reconstruction process.

The core component of the proposed model architecture consists of three main parts: an Attention-based Variational Autoencoder (VAE) or encoder, a Generator, and a Discriminator.

### C.1. Attention Block

The AttentionBlock is a key component in the methodology for 3D shape reconstruction. It enables the model to focus on important spatial features within the input image while preserving local information. The AttentionBlock is composed of self-attention mechanisms, which allow the model to capture dependencies between different parts of the input.

The AttentionBlock takes the input tensor and performs several operations to extract relevant features. It consists of a projection step to obtain query, key, and value vectors from the input. These vectors are then used to compute attention scores, which represent the importance of different parts of the input. The attention scores are normalized using a softmax function to obtain attention probabilities. These probabilities are applied to the value vectors to obtain a weighted sum, which represents the attended features. The attended features are then processed further, typically through additional projection and non-linear activation layers, to produce the final output of the AttentionBlock.

By incorporating AttentionBlocks into the model architecture, the methodology can effectively capture the relationships and dependencies within the input data, allowing for more accurate and precise 3D shape reconstruction. The attention mechanism helps the model to focus on important spatial features while preserving local details, resulting in improved performance compared to approaches that do not utilize attention.

### C.2. Variational Encoder

The encoder is composed of three convolutional blocks, with two attention blocks inserted in between. The attention blocks allow the model to focus on important spatial features within the input image while preserving local information. Each convolutional block consists of a convolutional layer, followed by max pooling and ReLU activation, enabling the extraction of meaningful features from the image.

After the convolutional blocks, the encoder reduces the extracted features to a flattened vector, representing the latent representation of the input image. This latent representation captures the essence of the 3D shape and serves as a compact representation for the subsequent stages of the model.

### C.3. Generator and Discriminator

The generator takes the latent representation as input and samples from a normal distribution with mean and variance learned from the latent vector. Then it aims to generate 3D voxel representations that closely resemble the true 3D shape. It employs upsampling layers, along with convolutional layers and ReLU activation, to gradually increase the spatial dimensions and refine the generated voxel representations.

The discriminator plays a crucial role in distinguishing between real and generated 3D voxel representations. It consists of convolutional layers, followed by max pooling and LeakyReLU activation, to learn discriminative features from the voxel data. The discriminator is trained to classify whether a given voxel representation is real or generated.

By training the model using a combination of the VAE and GAN objectives, we aim to enhance the quality of the generated 3D shapes and improve their resemblance to the real shapes. The VAE loss encourages the model to learn a compact and meaningful latent representation, while the GAN loss promotes the generation of realistic 3D shapes that can deceive the discriminator.

Overall, the combination of attention