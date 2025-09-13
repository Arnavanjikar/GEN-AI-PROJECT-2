Quick tour
Let's have a quick look at the 🤗 Transformers library features. 

The library downloads pretrained models for Natural Language Understanding (NLU) tasks, such as analyzing the sentiment of a text, and Natural Language Generation (NLG), such as completing a prompt with new text or translating in another language.

First we will see how to easily leverage the pipeline API to quickly use those pretrained models at inference. Then, we will dig a little bit more and see how the library gives you access to those models and helps you preprocess your data.
Getting started on a task with a pipeline
The easiest way to use a pretrained model on a given task is to use pipeline. 🤗 Transformers provides the following tasks out of the box:

Sentiment analysis: is a text positive or negative?
Text generation (in English): provide a prompt and the model will generate what follows.
Name entity recognition (NER): in an input sentence, label each word with the entity it represents (person, place, etc.)
Question answering: provide the model with some context and a question, extract the answer from the context.
Filling masked text: given a text with masked words (e.g., replaced by [MASK]), fill the blanks.
Summarization: generate a summary of a long text.
Translation: translate a text in another language.
Feature extraction: return a tensor representation of the text.
When typing this command for the first time, a pretrained model and its tokenizer are downloaded and cached. We will look at both later on, but as an introduction the tokenizer's job is to preprocess the text for the model, which is then responsible for making predictions. The pipeline groups all of that together, and post-process the predictions to make them readable. For instance:

That's encouraging! You can use it on a list of sentences, which will be preprocessed then fed to the model as a batch, returning a list of dictionaries like this one:
You can see the second sentence has been classified as negative (it needs to be positive or negative) but its score is fairly neutral.

By default, the model downloaded for this pipeline is called "distilbert-base-uncased-finetuned-sst-2-english". We can look at its model page to get more information about it. It uses the DistilBERT architecture and has been fine-tuned on a dataset called SST-2 for the sentiment analysis task.

Let's say we want to use another model; for instance, one that has been trained on French data. We can search through the model hub that gathers models pretrained on a lot of data by research labs, but also community models (usually fine-tuned versions of those big models on a specific dataset). Applying the tags "French" and "text-classification" gives back a suggestion "nlptown/bert-base-multilingual-uncased-sentiment". Let's see how we can use it.

You can directly pass the name of the model to use to pipeline

This classifier can now deal with texts in English, French, but also Dutch, German, Italian and Spanish! You can also replace that name by a local folder where you have saved a pretrained model (see below). You can also pass a model object and its associated tokenizer.

We will need two classes for this. The first is AutoTokenizer, which we will use to download the tokenizer associated to the model we picked and instantiate it. The second is AutoModelForSequenceClassification (or TFAutoModelForSequenceClassification if you are using TensorFlow), which we will use to download the model itself. Note that if we were using the library on an other task, the class of the model would change. The task summary tutorial summarizes which class is used for which task.



Now, to download the models and tokenizer we found previously, we just have to use the AutoModelForSequenceClassification.from_pretrained method (feel free to replace model_name by any other model from the model hub):
If you don't find a model that has been pretrained on some data similar to yours, you will need to fine-tune a pretrained model on your data. We provide example scripts to do so. Once you're done, don't forget to share your fine-tuned model on the hub with the community, using this tutorial.




Let's now see what happens beneath the hood when using those pipelines. As we saw, the model and tokenizer are created using the from_pretrained method

Using the tokenizer
We mentioned the tokenizer is responsible for the preprocessing of your texts. First, it will split a given text in words (or part of words, punctuation symbols, etc.) usually called tokens. There are multiple rules that can govern that process (you can learn more about them in the tokenizer summary), which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules as when the model was pretrained.

The second step is to convert those tokens into numbers, to be able to build a tensor out of them and feed them to the model. To do this, the tokenizer has a vocab, which is the part we download when we instantiate it with the from_pretrained method, since we need to use the same vocab as when the model was pretrained.

This returns a dictionary string to list of ints. It contains the ids of the tokens, as mentioned before, but also additional arguments that will be useful to the model. Here for instance, we also have an attention mask that the model will use to have a better understanding of the sequence:

You can pass a list of sentences directly to your tokenizer. If your goal is to send them through your model as a batch, you probably want to pad them all to the same length, truncate them to the maximum length the model can accept and get tensors back. You can specify all of that to the tokenizer

The padding is automatically applied on the side expected by the model (in this case, on the right), with the padding token the model was pretrained with. The attention mask is also adapted to take the padding into account

Using the model
Once your input has been preprocessed by the tokenizer, you can send it directly to the model. As we mentioned, it will contain all the relevant information the model needs. If you're using a TensorFlow model, you can pass the dictionary keys directly to tensors, for a PyTorch model, you need to unpack the dictionary by adding **

In 🤗 Transformers, all outputs are tuples (with only one element potentially). Here, we get a tuple with just the final activations of the model.
The model can return more than just the final activations, which is why the output is a tuple. Here we only asked for the final activations, so we get a tuple with one element.

NOTE: All 🤗 Transformers models (PyTorch or TensorFlow) return the activations of the model before the final activation function (like SoftMax) since this final activation function is often fused with the loss.

If you have labels, you can provide them to the model, it will return a tuple with the loss and the final activations.

Models are standard torch.nn.Module or tf.keras.Model so you can use them in your usual training loop. 🤗 Transformers also provides a Trainer (or TFTrainer if you are using TensorFlow) class to help with your training (taking care of things such as distributed training, mixed precision, etc.). See the training tutorial for more details.

NOTE: Pytorch model outputs are special dataclasses so that you can get autocompletion for their attributes in an IDE. They also behave like a tuple or a dictionary (e.g., you can index with an integer, a slice or a string) in which case the attributes not set (that have None values) are ignored.

You can then load this model back using the AutoModel.from_pretrained method by passing the directory name instead of the model name. One cool feature of 🤗 Transformers is that you can easily switch between PyTorch and TensorFlow: any model saved as before can be loaded back either in PyTorch or TensorFlow. If you are loading a saved PyTorch model in a TensorFlow model, use TFAutoModel.from_pretrained .

Accessing the code
The AutoModel and AutoTokenizer classes are just shortcuts that will automatically work with any pretrained model. Behind the scenes, the library has one model class per combination of architecture plus class, so the code is easy to access and tweak if you need to.

In our previous example, the model was called "distilbert-base-uncased-finetuned-sst-2-english", which means it's using the DistilBERT architecture. As AutoModelForSequenceClassification (or TFAutoModelForSequenceClassification if you are using TensorFlow) was used, the model automatically created is then a DistilBertForSequenceClassification. You can look at its documentation for all details relevant to that specific model, or browse the source code. This is how you would directly instantiate model and tokenizer without the auto magic:

Customizing the model
If you want to change how the model itself is built, you can define your custom configuration class. Each architecture comes with its own relevant configuration (in the case of DistilBERT, DistilBertConfig) which allows you to specify any of the hidden dimension, dropout rate, etc. If you do core modifications, like changing the hidden size, you won't be able to use a pretrained model anymore and will need to train from scratch. You would then instantiate the model directly from this configuration.

Here we use the predefined vocabulary of DistilBERT (hence load the tokenizer with the DistilBertTokenizer.from_pretrained method) and initialize the model from scratch (hence instantiate the model from the configuration instead of using the DistilBertForSequenceClassification.from_pretrained method).

For something that only changes the head of the model (for instance, the number of labels), you can still use a pretrained model for the body. For instance, let's define a classifier for 10 different labels using a pretrained body. We could create a configuration with all the default values and just change the number of labels, but more easily, you can directly pass any argument a configuration would take to the from_pretrained method and it will update the default configuration with it:






