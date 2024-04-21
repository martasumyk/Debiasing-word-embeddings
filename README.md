# Debiasing word embeddings

### Problem statement 
Large Language Models (LLMs) generate outputs based on the datasets they've been trained on. Consequently, if these datasets contain stereotypes, the LLMs themselves will reflect and perpetuate those biases.

### Aim
Reduce the bias in the LLM using linear algebra techniques.

### Indicating and measuring the bias

In our research we are using [Microsoft phi-2 model](https://huggingface.co/microsoft/phi-2).

To indentify the bias, we prompted the model with "Woman are better than man in " and get the following answer:

```css
Women are better than men in many ways. They are more nurturing, more empathetic, and more compassionate. They are also better at multitasking and have better communication skills. These qualities make them better suited for leadership roles in the workplace.
 However, women still face many challenges in the workplace. They are often paid less than men for doing the same job, and they are underrepresented in leadership positions. This is why it is important to promote women's leadership and provide them with the support they need to succeed. One way to promote women's leadership is to provide them with mentorship and networking opportunities. This can help them build relationships with other women in their field and learn from their experiences. It can also help them develop the skills they need to succeed in leadership roles.
 Another way to promote women's leadership is to provide them with training and development opportunities. This can help them build their skills and knowledge and prepare them for leadership roles.
```

We can see that the model poseses gender-biased results.

To measure the bias we use the cosine similarity, which is defined as:

\cos(\vec{u}, \vec{v}) = \frac{\vec{u}\vec{v}}{\lVert \vec{u} \rVert \lVert \vec{v} \rVert}








