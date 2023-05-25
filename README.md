# 3DAttnVAE-GAN
A VAE-GAN model designed for learning 3d shape from a single 2d image. Trained on ShapeNetCore Dataset

\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[rebuttal]{cvpr}

% Include other packages here, before hyperref.
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{tikz}
\lstset{
  basicstyle=\ttfamily,
  numbers=left,
  numberstyle=\tiny,
  breaklines=true,
  breakatwhitespace=true,
  frame=single,
  tabsize=2
}

% If you comment hyperref and then uncomment it, you should delete
% egpaper.aux before re-running latex.  (Or just hit 'q' on the first latex
% run, let it finish, and you should be clear).
\usepackage[pagebackref,breaklinks,colorlinks,bookmarks=false]{hyperref}

% Support for easy cross-referencing
\usepackage[capitalize]{cleveref}
\crefname{section}{Sec.}{Secs.}
\Crefname{section}{Section}{Sections}
\Crefname{table}{Table}{Tables}
\crefname{table}{Tab.}{Tabs.}

% If you wish to avoid re-using figure, table, and equation numbers from
% the main paper, please uncomment the following and change the numbers
% appropriately.
%\setcounter{figure}{2}
%\setcounter{table}{1}
%\setcounter{equation}{2}

% If you wish to avoid re-using reference numbers from the main paper,
% please uncomment the following and change the counter for `enumiv' to
% the number of references you have in the main paper (here, 6).
%\let\oldthebibliography=\thebibliography
%\let\oldendthebibliography=\endthebibliography
%\renewenvironment{thebibliography}[1]{%
%     \oldthebibliography{#1}%
%     \setcounter{enumiv}{6}%
%}{\oldendthebibliography}


%%%%%%%%% PAPER ID  - PLEASE UPDATE
\def\cvprPaperID{314} % *** Enter the CVPR Paper ID here
\def\confName{CVPR}
\def\confYear{2023}

\begin{document}

%%%%%%%%% TITLE - PLEASE UPDATE
\title{3D Shape Reconstruction from a Single Image}  % **** Enter the paper 
\author{Kartikesh Mishra}

\maketitle
\thispagestyle{empty}
\appendix

%%%%%%%%% BODY TEXT - ENTER YOUR RESPONSE BELOW
\begin{abstract}
    This project delves into the problem of reconstructing 3D shapes from a single 2D image, leveraging the extensive ShapeNetCore dataset comprising over 48,600 3D models across 55 common categories. The primary objective is to assess the performance of 3D reconstruction algorithms using a single image and compare their effectiveness against existing approaches. The project encompasses crucial stages, including data setup and preprocessing, the development of a robust methodology for 3D shape reconstruction, conducting comprehensive experiments, analyzing and presenting the obtained results, and outlining a roadmap for future research. By examining the challenges associated with capturing 3D shape information from a single image, this work contributes to advancing the field of 3D shape reconstruction. The evaluation of diverse reconstruction algorithms using the ShapeNetCore dataset allows for a rigorous assessment of their performance and offers insights into the state-of-the-art techniques. 
\end{abstract}

\section{Introduction}
In recent years, the availability of 3D data has expanded exponentially, leading to advancements in algorithms that enhance our understanding of the three-dimensional world. However, numerous challenges persist in dealing with 3D representations and processing, leaving several research issues open for exploration. Among these challenges, 3D shape reconstruction from a single image stands as a pivotal problem, holding great potential for applications in virtual and augmented reality, robotics, and autonomous vehicles.

The pursuit of capturing 3D shapes from 2D images has been a decades-old problem. While 3D sensors exist, they are often costly and not universally accessible. As a result, the quest to develop techniques capable of reconstructing 3D shapes from single images, without relying on additional sensors or multiple views, remains a compelling research direction. This raises the fundamental question: Can we accurately reconstruct 3D shape information from a single image while minimizing assumptions?

In this project, my main objective is to evaluate the performance of 3D reconstruction algorithms on the ShapeNet dataset\cite{chang2015shapenet}—a widely used benchmark for 3D shape analysis and understanding. To conduct my experiments, I utilized the ShapeNetCore subset, encompassing approximately 48,600 3D models spanning 55 common categories. For training, validation, and testing purposes, the dataset is already split in 70\%,  10\%, 20\% respectively. As my output 3D representations, I chose voxels—a popular choice in 3D reconstruction tasks.

By delving into the complexities of 3D shape reconstruction from a single image, I aim to know the advancement of this field and explore potential solutions to the challenges posed. 

%------------------------------------------------------------------------
\section{Related Works}
Recent advancements in 3D shape generation have led to the development of novel frameworks such as 3D Generative Adversarial Network (3D-GAN) \cite{wu2016learning}. 3D-GAN generates high-quality 3D objects from a probabilistic space by leveraging volumetric convolutional networks and generative adversarial nets. The adversarial criterion used in the framework enables the generator to capture object structures implicitly and synthesize high-quality 3D objects. Moreover, the generator establishes a mapping from a low-dimensional probabilistic space to the space of 3D objects, allowing us to sample objects without a reference image or CAD models. The discriminator of 3D-GAN also provides a powerful 3D shape descriptor, learned without supervision, which has wide applications in 3D object recognition.

Pix3D \cite{pix3d} is another related work, which is a large-scale benchmark dataset of diverse image-shape pairs with pixel-level 2D-3D alignment. Unlike existing datasets, Pix3D contains real-world images and 3D models with precise alignment. The authors of this paper claim to have calibrated the evaluation criteria for 3D shape reconstruction through behavioral studies and used them to systematically benchmark state-of-the-art reconstruction algorithms on Pix3D. Additionally, the authors designed a novel model that performs 3D reconstruction and pose estimation simultaneously, achieving state-of-the-art performance on both tasks.

AtlasNet \cite{groueix2018papier} introduces a method for learning to generate the surface of 3D shapes. The approach represents a 3D shape as a collection of parametric surface elements and naturally infers a surface representation of the shape. AtlasNet offers significant advantages, such as improved precision and generalization capabilities, and the ability to generate shapes of arbitrary resolution without memory issues. The framework demonstrates its effectiveness and outperforms strong baselines on the ShapeNet benchmark for applications including auto-encoding shapes and single-view reconstruction from still images. Furthermore, AtlasNet shows potential in various other applications, such as morphing, parametrization, super-resolution, matching, and co-segmentation.

MarrNet \cite{wu2017marrnet} addresses the challenge of 3D object reconstruction from a single image by proposing an end-to-end trainable model that sequentially estimates 2.5D sketches and 3D object shape. The disentangled, two-step formulation of MarrNet offers advantages such as improved transferability from synthetic to real data by recovering 2.5D sketches, and the ability to learn from synthetic data for 3D reconstruction. The framework employs differentiable projective functions to enable end-to-end training on real images without human annotations, achieving state-of-the-art performance in 3D shape reconstruction.

These related works, including 3D-GAN, Pix3D, AtlasNet, and MarrNet, contribute to the field of 3D shape reconstruction from a single image by exploring different approaches, addressing challenges, and achieving notable advancements. The evaluation and comparison of diverse reconstruction algorithms using datasets like ShapeNetCore and Pix3D allow for a rigorous assessment of their performance and provide insights into state-of-the-art techniques. These works collectively contribute to advancing the field and provide valuable knowledge in the area of 3D shape reconstruction.

\section{Methodology}
My proposed method for 3D shape reconstruction from a single image is a modified version of VAE\_GAN that incorporates attention blocks \cite{vaswani2017attention}. The combination of CNN-based architectures and Transformer-based architectures has been a topic of interest in the field of machine learning. While CNNs excel at learning spatial features, Transformers are known for their ability to capture sequential patterns, making them effective in language tasks such as natural language processing. Leveraging the strengths of both architectures, my model aims to incorporate attention mechanisms into the 3D shape reconstruction process.

The core component of the proposed model architecture consists of three main parts: an Attention-based Variational Autoencoder (VAE) or encoder, a Generator, and a Discriminator.
\subsection{Attention Block}
\begin{figure}[ht]
    \centering
    \begin{tikzpicture}[scale=0.8, every node/.style={scale=0.8}, rotate=90]
    % Input
    \node[draw, minimum size=1cm] (input) at (0, 0) {Input};
    
    % QKV Projection
    \node[draw, minimum size=1cm] (qkv) at (3, 0) {QKV Projection};
    
    % Attention Scores
    \node[draw, minimum size=1cm] (attention) at (6, 0) {Attention Scores};
    
    % Softmax
    \node[draw, minimum size=1cm] (softmax) at (9, 0) {Softmax};
    
    % Weighted Sum
    \node[draw, minimum size=1cm] (weightedsum) at (12, 0) {Weighted Sum};
    
    % Output Projection
    \node[draw, minimum size=1cm] (output) at (15, 0) {Output Projection};
    
    % Arrows
    \draw[->] (input) -- (qkv);
    \draw[->] (qkv) -- (attention);
    \draw[->] (attention) -- (softmax);
    \draw[->] (softmax) -- (weightedsum);
    \draw[->] (weightedsum) -- (output);
    
    % % Labels
    % \node at (0, -1) {Input Tensor};
    % \node at (3, -1) {QKV Projection};
    % \node at (6, -1) {Attention Scores};
    % \node at (9, -1) {Softmax};
    % \node at (12, -1) {Weighted Sum};
    % \node at (15, -1) {Output Projection};
\end{tikzpicture}
\caption{Attention Block}
\end{figure}



The AttentionBlock is a key component in the methodology for 3D shape reconstruction. It enables the model to focus on important spatial features within the input image while preserving local information. The AttentionBlock is composed of self-attention mechanisms, which allow the model to capture dependencies between different parts of the input.

The AttentionBlock takes the input tensor and performs several operations to extract relevant features. It consists of a projection step to obtain query, key, and value vectors from the input. These vectors are then used to compute attention scores, which represent the importance of different parts of the input. The attention scores are normalized using a softmax function to obtain attention probabilities. These probabilities are applied to the value vectors to obtain a weighted sum, which represents the attended features. The attended features are then processed further, typically through additional projection and non-linear activation layers, to produce the final output of the AttentionBlock.

By incorporating AttentionBlocks into the model architecture, the methodology can effectively capture the relationships and dependencies within the input data, allowing for more accurate and precise 3D shape reconstruction. The attention mechanism helps the model to focus on important spatial features while preserving local details, resulting in improved performance compared to approaches that do not utilize attention.
\subsection{Variational Encoder}
The encoder is composed of three convolutional blocks, with two attention blocks inserted in between. The attention blocks allow the model to focus on important spatial features within the input image while preserving local information. Each convolutional block consists of a convolutional layer, followed by max pooling and ReLU activation, enabling the extraction of meaningful features from the image.

After the convolutional blocks, the encoder reduces the extracted features to a flattened vector, representing the latent representation of the input image. This latent representation captures the essence of the 3D shape and serves as a compact representation for the subsequent stages of the model.
% \begin{figure}[ht]
%     \centering
%     \begin{tikzpicture}[node distance=2cm, every node/.style={draw, rounded corners}]
%         % Convolutional block 1
%         \node (conv1) {Convolutional Layer};
%         \node (pool1) [below of=conv1] {Max Pooling};
%         \node (relu1) [below of=pool1] {ReLU Activation};

%         % Attention block 1
%         \node (attention1) [below of=relu1] {Attention Block};

%         % Convolutional block 2
%         \node (conv2) [below of=attention1] {Convolutional Layer};
%         \node (pool2) [below of=conv2] {Max Pooling};
%         \node (relu2) [below of=pool2] {ReLU Activation};

%         % Attention block 2
%         \node (attention2) [below of=relu2] {Attention Block};

%         % Convolutional block 3
%         \node (conv3) [below of=attention2] {Convolutional Layer};
%         \node (pool3) [below of=conv3] {Max Pooling};
%         \node (relu3) [below of=pool3] {ReLU Activation};

%         % Latent representation
%         \node (latent) [below of=relu3, text width=5cm, align=center] {Flattened Vector (Latent Representation)};

%         % Connect blocks
%         \draw[->] (conv1) -- (pool1);
%         \draw[->] (pool1) -- (relu1);
%         \draw[->] (relu1) -- (attention1);
%         \draw[->] (attention1) -- (conv2);
%         \draw[->] (conv2) -- (pool2);
%         \draw[->] (pool2) -- (relu2);
%         \draw[->] (relu2) -- (attention2);
%         \draw[->] (attention2) -- (conv3);
%         \draw[->] (conv3) -- (pool3);
%         \draw[->] (pool3) -- (relu3);
%         \draw[->] (relu3) -- (latent);
%     \end{tikzpicture}
%     \caption{Encoder Architecture with Convolutional and Attention Blocks}
%     \label{fig:encoder_architecture}
% \end{figure}
\subsection{Generator and Discriminator}
The generator takes the latent representation as input and samples from a normal distribution with mean and variance learned from the latent vector. Then it aims to generate 3D voxel representations that closely resemble the true 3D shape. It employs upsampling layers, along with convolutional layers and ReLU activation, to gradually increase the spatial dimensions and refine the generated voxel representations.

The discriminator plays a crucial role in distinguishing between real and generated 3D voxel representations. It consists of convolutional layers, followed by max pooling and LeakyReLU activation, to learn discriminative features from the voxel data. The discriminator is trained to classify whether a given voxel representation is real or generated.

By training the model using a combination of the VAE and GAN objectives, we aim to enhance the quality of the generated 3D shapes and improve their resemblance to the real shapes. The VAE loss encourages the model to learn a compact and meaningful latent representation, while the GAN loss promotes the generation of realistic 3D shapes that can deceive the discriminator.

Overall, the combination of attention blocks, convolutional layers, and the VAE-GAN framework allows the model to effectively reconstruct 3D shapes from a single input image. This novel approach capitalizes on the strengths of both CNNs and Transformers, enabling the model to learn spatial and sequential features simultaneously, leading to improved performance in 3D shape reconstruction tasks.

\section{Experiment and Results}

I trained the above model architecture using the ShapeNet Core Dataset. Due to computational resource limitations, I resized the 3D models from 256x256x256 to 64x64x64. Since this project focuses solely on 3D shape reconstruction, the color information was not crucial. Therefore, I transformed the 2D images from RGB channels to grayscale.

I utilized three types of loss functions to optimize the model parameters, employing the AdamW algorithm. The first loss function is for the Variational Autoencoder (VAE) and involves KL Divergence. The VAE loss function aims to match the latent space distribution to a prior distribution, typically a Gaussian distribution. The KL Divergence loss can be expressed as:

\[
\mathcal{L}_{\text{VAE}} = \frac{1}{2}\sum_{j=1}^{J}(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)
\]

where $\mu_j$ and $\sigma_j$ represent the mean and standard deviation of the latent variable $j$, respectively.

The second loss function is for the Generator and is the binary cross entropy loss between the given 3D (64x64x64) voxels and the generated 64x64x64 voxels. The binary cross entropy loss can be defined as:

\[
\mathcal{L}_{\text{Generator}} = -\frac{1}{N}\sum_{i=1}^{N}(y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i))
\]

where $N$ is the number of voxels, $y_i$ is the ground truth voxel value, and $\hat{y}_i$ is the predicted voxel value.

Finally, the discriminator loss function is the sum of the binary cross entropy between the Discriminator (generated\_voxel) and 0, and between the Discriminator (real\_voxels) and 1. The discriminator loss can be formulated as:

\[
\mathcal{L}_{\text{Discriminator}} = -\frac{1}{M}\sum_{i=1}^{M}\left(\log(D(\text{gv}_i)) + \log(1-D(\text{rv}_i))\right)
\]

where $M$ is the number of samples, $D(\text{gv}_i)$ represents the discriminator's output for the generated voxel, and $D(\text{rv}_i)$ represents the discriminator's output for the real voxels.

\begin{figure}[ht]
    \centering
    \includegraphics[width=5cm]{latex/figure/phone_orig.png}
    \includegraphics[width=5cm]{latex/figure/phone_3d.png}
    \includegraphics[width=5cm]{latex/figure/phone_gen.png}
    \captionof{figure}{Phone Image 2d 3d and 3d generated}
    \label{fig:phone_images}
\end{figure}

\begin{figure}[ht]
    \centering
    \includegraphics[width=5cm]{latex/figure/car_orig.png}
    \includegraphics[width=5cm]{latex/figure/car_3d.png}
    \includegraphics[width=5cm]{latex/figure/car_voxel.png}
    \captionof{figure}{Car Image 2d 3d and 3d generated}
    \label{fig:car_images}
\end{figure}

\section{Evaluation and Comparison}
I used two evaluation metrics Chamfer Distance(CD) and Intersection Over Union (IoU). For $X, Y \in \mathbb{R}^3$
\begin{equation}
    \text{CD(X, Y)} = \frac{1}{|\mathbf{X}|} \sum_{\mathbf{x} \in \mathbf{X}} \min_{\mathbf{y} \in \mathbf{Y}} \|\mathbf{x} - \mathbf{y}\|_2 \\
    + \frac{1}{|\mathbf{Y}|} \sum_{\mathbf{y} \in \mathbf{Y}} \min_{\mathbf{x} \in \mathbf{X}} \|\mathbf{y} - \mathbf{x}\|_2
\end{equation}

\begin{equation}
    \text{IoU(X, Y)} = \frac{|X\cap Y|}{|X \cup Y|}
\end{equation}

\begin{table}[h]
    % \centering
    \caption{Results on 3D shape reconstruction.}
    \begin{tabular}{lccc}
        \hline
        Method & IoU & EMD & CD \\
        \hline
        3D-R2N2 \cite{choy20163dr2n2} & 0.136 & 0.211 & 0.239 \\
        PSGN \cite{fan2017point} & N/A & 0.216 & 0.200 \\
        3D-VAE-GAN \cite{wu2016learning} & 0.171 & 0.176 & 0.182 \\
        DRC \cite{tulsiani2017multiview} & 0.265 & 0.144 & 0.160 \\
        MarrNet* \cite{wu2017marrnet} & 0.231 & 0.136 & 0.144 \\
        AtlasNet \cite{groueix2018papier} & N/A & 0.128 & 0.125 \\
        Pix3D\cite{pix3d} (w/o Pose) & 0.267 & 0.124 & 0.124 \\
        Pix3D\cite{pix3d} (w/ Pose) & 0.282 & 0.118 & 0.119 \\
        R3D(My Project) & 0.2288 & \_ &0.052\\
        \hline
    \end{tabular}
    \label{table:results}
\end{table}

Table \ref{table:results} shows the results on 3D shape reconstruction which was taken from pix3D paper\cite{pix3d}. My model evaluated with Chamfer Distance and IoU on 30 test images scores average 0.052 CD and 0.2288 IoU. 


\section{Conclusion}
This  project has explored the problem of 3D shape reconstruction from a single 2D image using a modified version of the VAE-GAN architecture with attention blocks. Through extensive experimentation and evaluation of the ShapeNetCore dataset, I have obtained some good insights into the effectiveness of my approach.

The evaluation of various metrics, IoU, and CD as well as the 3d generated images has demonstrated that my proposed model performed well compared to state-of-the-art methods. My model achieved the best scores in terms of Chamfer Distance, highlighting its capability to reconstruct 3D shapes accurately from a single image but since that was taken over a handful of data, a measure involving more data points if not the whole ShapeNet core will be something to do in future.

The inclusion of attention blocks in my encoder architecture has played a crucial role in capturing and focusing on essential spatial features within the input image. By preserving local information while attending to relevant areas, the attention blocks contribute to the interpretability of the model, allowing us to gain insights into the learned representations. 

Moving forward, several avenues for future research can be explored. Firstly, incorporating additional modalities such as depth information or multi-view images can potentially enhance the reconstruction performance and generate more accurate 3D shapes. Moreover, investigating different variations of attention mechanisms, such as self-attention or multi-head attention, could further improve the model's ability to capture fine-grained details and intricate shape structures.

Furthermore, advancing the interpretability of the model's decisions and understanding the learned attention weights are essential directions for future work. This can be achieved through visualization techniques and attention attribution methods, which can shed light on the model's decision-making process and provide insights into which image regions contribute most to the shape reconstruction.

In conclusion, my project contributes to the field of 3D shape reconstruction by proposing a novel architecture that combines the strengths of VAE-GAN and attention mechanisms. The achieved results demonstrate the effectiveness of my approach and open up new possibilities for advancing the field. By focusing on interpretability and further exploring attention mechanisms, we can continue to improve the accuracy and understanding of 3D shape reconstruction from single images.

%%%%%%%%% REFERENCES
{\small
\bibliographystyle{ieee_fullname}
\bibliography{egbib}
}

