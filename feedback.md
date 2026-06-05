## Assignment 1

```
Perplexity: no worries about the implementation, it looks completely normal. The perplexity values mentioned in the instructions are a bit higher than what most people got, simply because I didn't expect people to tune so carefully... (and of course it also depends on choices such as the vocabulary size).

About the vocab size: you can also reason in terms of the frequency of each token. If your vocab size is large, some of these embeddings will have very few updates.
```

## Assignment 2

```
Same answer as for assignment 1: you just did a more careful job than I did myself and than I expected students to do...

With this data size (and probably your model is rather small, but I can't see your choices in that respect), our expectation is basically that the model is able to "learn English" but the text itself will typically be rather nonsensical.
```

## Assignment 3

```
On a high level this looks more or less as expected, as far as I can tell. For more detailed feedback, you will have to ask more specific questions.
```

## Assignment 4

```
As I wrote for the previous assignment, we will provide feedback in case you are asking about specific points.

Anyway, in your case I think it could be useful to look at the confusion matrices, since it seems to me that your implementation has a strong tendency to say "yes". (And I guess either the confusion matrices or the precision/recall values have been computed backwards? The recall values for "no" are close to 1.)
```

## How I used the feedback

For Assignments 1 to 3, everything seemed OK so I did not make any changes based on the feedback. For Assignment 4 I believe the issue is that I have `labels=labels` for the matrix plot and `target_names=labels` for the report (recall value etc.). I have not changed this in the current commit, because I need to use a large GPU to fit Qwen3-8B and it takes a while to rerun it all. The results are there, yes/no should just be swapped in the report. 
