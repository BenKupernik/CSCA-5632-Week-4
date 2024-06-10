#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:20:46 2024

@author: benkupernk
"""
from collections import Counter 
import matplotlib.pyplot as plt
def word_plotter(df, filter_words = None, cat = 'all', num_words = 10, show_words = False):
    if cat != 'all':
        df_temp = df[df.Category == cat]
    else:
        df_temp = df
        
    words = df_temp.Text.to_list()
    words = ' '.join(words)
    words = words.split(' ')
    word_count = Counter(words)
    
    # delete the entry if the key is a filter word
    if filter_words is not None:
        for filter_word in filter_words:
            del word_count[filter_word] 
                       
    # the most common function returns a list and its easier to plot as a dict 
    top_words = dict([(word, count) for word, count in word_count.most_common(num_words)])
    
    # make a coulmn for cumulative words
    total_words = sum(word_count.values())
    count_so_far = 0
    percent = []
    for count in top_words.values():
        count_so_far += count
        percent.append((count_so_far/total_words)*100)
        
    fig, ax1 = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(8)
    plt.bar(list(top_words.keys()), list(top_words.values()))
    plt.xticks(rotation=90)
    
    ax2 = plt.twinx()
    ax2.plot(list(top_words.keys()), percent, color='y', label='Cumulative Words')
    
    plt.title(cat)
    ax1.set_xlabel('Most Commen Words')
    ax1.set_ylabel('Word Count')
    ax2.set_ylabel('Cumulative Percent')
    
    if show_words:
        print('The top %s words for %s articles are %s' % (num_words, cat, list(top_words.keys())))
    plt.show()
    
    
    
    
def nmf_metrics(df, nmf, w, tfidf_vectorizer, plot=False, true_labels = None):
    """Prints out the accuracy and optionally makes a plot displaying the top 10 words and categories for the nmf model results"""
    # use the label fiting function from week 2 assign catagories to each document and calculate the accuracy
    if true_labels is None:
        best_label, acc = label_permute_compare(df.Category, get_predictions(w))
    else:
        best_label, acc = label_permute_compare(true_labels, get_predictions(w))
    tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    if plot:
        # make a plot showing each catagory and its top words
        plot_top_words(nmf, tfidf_feature_names, 10, "NMF", best_label)
    return acc

def get_predictions(w):
    '''Once you have your w matrix this will return which column had the highest value
    ie which category the article is predicted to be'''
    predictions = np.argmax(w, axis=1)
    return predictions

def label_permute_compare(true_, predicted, n=5):
    '''Gets the combination of labels that results in the highest accuracy'''
    perms = itertools.permutations(['business', 'tech', 'politics', 'sport', 'entertainment'])
    
    acc = 0.0
    best_order = 0
    for perm in perms:
        pred = [perm[i] for i in predicted]
        pred_acc = sum([1 for x, y, in zip(list(true_), pred) if x == y]) / len(true_)
        if pred_acc > acc:
            acc = pred_acc
            best_order = perm
            
    return best_order, acc

def show_words_for_topics(topic, words,  num_words = 10):
    '''Takes the H matrix from nmf and a matrix of the words in the text. Picks the num_words with the highest 
    score for each category and returns them'''
    return np.apply_along_axis(lambda x: words[(np.argsort(-x))[:num_words]], 1, topic)

def plot_top_words(model, feature_names, n_top_words, title, subtitles):
    '''Takes the nmf model, a list of category names and some styling information for the graph.
    It then plots a horizontal bar chart for each category. Modified from the sklearn example here
    https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html'''
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(subtitles[topic_idx], fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
        