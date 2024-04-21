# Debiasing word embeddings

### Problem statement 
Large Language Models (LLMs) generate outputs based on the datasets they've been trained on. Consequently, if these datasets contain stereotypes, the LLMs themselves will reflect and perpetuate those biases.

### Aim
Reduce the bias in the LLM using linear algebra techniques.

### Indicating and measuring the bias

In our research we are using [Microsoft phi-2 model](https://huggingface.co/microsoft/phi-2).

To indentify the bias, we prompted the model with "Woman are better than man in " and get the following answer:

_Women are better than men in many ways. They are more nurturing, more empathetic, and more compassionate. They are also better at multitasking and have better communication skills. These qualities make them better suited for leadership roles in the workplace.
 However, women still face many challenges in the workplace. They are often paid less than men for doing the same job, and they are underrepresented in leadership positions. This is why it is important to promote women's leadership and provide them with the support they need to succeed. One way to promote women's leadership is to provide them with mentorship and networking opportunities. This can help them build relationships with other women in their field and learn from their experiences. It can also help them develop the skills they need to succeed in leadership roles.
 Another way to promote women's leadership is to provide them with training and development opportunities. This can help them build their skills and knowledge and prepare them for leadership roles._


We can see that the model poseses gender-biased results.

To measure the bias we use the cosine similarity metrics, which is defined as:

$$ \cos(\vec{u}, \vec{v}) = \frac{\vec{u}\vec{v}}{\lVert \vec{u} \rVert \lVert \vec{v} \rVert}, $$

where $\vec{u}\vec{v}$ denotes the inner product of the vectors $\vec{u}$ and $\vec{v}$ and $\lVert \vec{u} \rVert$, $\lVert \vec{v} \rVert$ denote the norms of the vectors $\vec{u}$ and $\vec{v}$ respectively.

The measure of bias itself is defined as:

$$\mbox{difference} = |\cos(\overrightarrow{w}, \overrightarrow{he}) - \cos(\overrightarrow{w}, \overrightarrow{she})|$$

The smaller is the difference, the less bias we have between the words.

### Defining gender-neutral and gender-specific subspaces

Firstly, we define the subset of word vectors intended to be gender-neutral (like $\overrightarrow{business}$ or $\overrightarrow{sport}$) by $N \subset \mathbb{R}^d $ and the gender-defining words  vectors subset (e.g. $\overrightarrow{man}$, $\overrightarrow{wonam}$, $\overrightarrow{he}$, $\overrightarrow{she}$) by $G \subset \mathbb{R}^d $.
Before defining set $G$, we established gender-specific pairs of words:

| Female-defining words | Male-defining words |
|-----------------------|---------------------|
| aunt                  | uncle               |
| daughter              | son                 |
| female                | male                |
| girl                  | boy                 |
| her                   | his                 |
| lass                  | lad                 |
| miss                  | mr                  |
| mom                   | dad                 |
| mother                | father              |
| she                   | he                  |
| wife                  | husband             |
| woman                 | man                 |
| women                 | men                 |

Differences between word embeddings vectors reflect distinctions in contextual usage. Therefore, to define the subspace of gender-specific words $G$, we utilize $13$ vectors representing the differences between the embeddings of the word pairs listed above. Let $S = \{v_1, v_2, ..., v_{13}\}$.

### Dimensionality reduction

Given that the basis for the subset $G$ consists of $13$ vectors, we aim to reduce this dimensionality in our research. To achieve this, we use Principal Component Analysis (PCA).

We get the following plot for the Elbow method:

![image](https://github.com/martasumyk/Debiasing-word-embeddings/assets/116710765/cc82931e-28f1-4373-8c87-a42b37e3cb28)



At the point where $k=7$, it becomes evident from the graphical representation that further reduction does not yield significant decrease. Consequently, we have selected $k=7$ and extracted the first $7$ eigenvectors for subsequent analysis as the basis vectors for the space $G$. Denote them ${g_1, ..., g_7}$ As these vectors are eigenvectors of a symmetric matrix, they are pairwise orthogonal.

### Soft debias algorithm

To neutralize the bias component of a given word vector $\vec{w}$, we aim to project this vector onto the gender-neutral subspace $N$. Given that $N + G = \mathbb{R}^d$, the spaces $G$ and $N$ are orthogonal. Consequently, every vector $\vec{w} \in \mathbb{R}^d$ can be decomposed as:

$$\vec{w} = \vec{w_{G}} + \vec{w_{N}},$$
where $\vec{w_{N}}$ represents the projection of $\vec{w}$ onto the affine space $N$, and $\vec{w_{G}}$ signifies the projection onto $G$.

Therefore,
$$\vec{w_{N}} = \vec{w} - \vec{w_{G}}$$

Given that the basis vectors of $G$ are orthogonal, we can obtain $\vec{w_{G}}$ as the sum of the projections onto these basis vectors:

$$\vec{w_{G}} = \sum_{i=1}^{7}{\vec{w_{g_i}}}$$
where $\vec{w_{g_i}}$ denotes the projection of $\vec{w}$ onto the basis vector $g_i$.
The projection matrix onto the linear span of $g_i$ is defined as:

$$P_{g_i} = \frac{g_ig_i^T}{g_i^Tg_i}$$
So the overall projection is calculated as follows:

$$\vec{w_{G}} = \sum_{i=1}^{7}{P_{g_i}\vec w} = \sum_{i=1}^{7}{\frac{g_ig_i^T}{g_i^Tg_i} \vec w}$$
Therefore, the debiased vector looks as follows:

$$\vec{w_{N}} = \vec{w} - \sum_{i=1}^{7}{P_{g_i}\vec w} = \vec{w} - \sum_{i=1}^{7}{\frac{g_ig_i^T}{g_i^Tg_i}\vec w}$$


### Hard debias algorithm

After soft debiasing a vector $\vec{w}$, we have a projection of this vector onto the gender-neutral subspace $N$. That mean that the the projection of this vector onto gender affine space is $0$. We have removed bias from word itself, but there is one more step to make it better. Our gender defining words not necessarily are equally distant. In order to make that a case, we use equalization that will make our gender defining words equally distant to all words $\vec{w_{N}}$. We don't have to recalculate distance for every word, as in subspace $N^\perp$  our vectors are $0$. 
$$\mu := \frac{\sum_{w \in E} W}{E}$$
Where $\mu$ is average of the set $E$.
In this formula we take average of every equalization set (in our case pair).
$$\nu := \mu - \mu_G $$
Where $\mu_G$ is projection of $\mu$ onto gender defining subspace.
Then for every $\vec{w}$
$$\vec{w} := \nu + \sqrt{1 - ||\nu||^2} \frac{\vec{w}_G - \mu_G}{||\vec{w}_G - \mu_G||}$$
Where $\vec{w}_G$ projection on gender defining subspaces.
This operations will make our gender pair vectors equally distant from all words that was changed using soft debias. 


### Debiasing results: WEAT test

Central to the WEAT test are two key hypotheses:

Null Hypothesis ($H_0$): There exists no significant difference between the two sets of target words concerning their relative associations with the sets of attribute words.

Alternative Hypothesis ($H_1$): There exists a difference between the two sets of target words.

The test statistic is computed by normalizing the mean difference between the associations of the target sets with the attribute sets. This statistic provides a quantified measure of the separation between the distributions of associations.

$$ \frac{\mbox{mean}_{x \in X}s(x, A, B) - \mbox{mean}_{y \in Y}s(y, A, B)}{\sigma_{w \in X \cup Y}s(w, A, B)}$$

Let ${(X_i, Y_i)}$ represent all possible partitions of $X\cup Y$ into two sets of equal size. The $p$-value for the one-sided permutation test is derived from the probability that the statistic computed for the partitions exceeds the statistic for the original sets ($s(X, Y, A, B)$).
$$P_i{s(X_i, Y_i, A, B) > s(X, Y, A, B)}$$

We got the following results:

- `Without debiasing`: The obtained $p$-value of $0.023$ leads to the rejection of the null hypothesis. Consequently, we conclude that there is a significant difference between the two sets of target words regarding their relative associations with the attribute sets.

- `Soft debiasing`: The $p$-value of $0.24$ does not provide sufficient evidence to reject the null hypothesis. Therefore, we can notice that there is substantially less difference between the two sets of target words concerning their relative associations with the attribute sets.

- `Hard debiasing`: With a $p$-value of $1$, we fail to reject the null hypothesis, indicating that there is no difference between the two sets of target words in terms of their relative associations with the attribute sets.



