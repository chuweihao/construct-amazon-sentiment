import os
import csv
import argparse
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from tqdm import tqdm
from math import exp
from itertools import chain


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Input data - amazon reviews data .json.gz')
    parser.add_argument('-c', '--concepts', type=str,
                        default='~/data/data-concept/data-concept-instance-relations.txt')
    parser.add_argument('-p', '--positive_opinions', type=str,
                        default='~/data/opinion-lexicon-English/positive-words.txt')
    parser.add_argument('-n', '--negative_opinions', type=str,
                        default='~/data/opinion-lexicon-English/negative-words.txt')
    parser.add_argument('-o', '--out', type=str,
                        help='Output directory, final output including: rating.txt, sentiment.txt')
    parser.add_argument('--num_top_freq_aspect', type=int, default=2000)
    parser.add_argument('--num_top_corr_aspect', type=int, default=500)
    return parser.parse_args()


def parse_line(line, n_grams=[1, 2, 3, 4]):
    tokens = word_tokenize(line)
    results = []
    for i in n_grams:
        if i == 1:
            results += tokens
        elif len(tokens) >= i:
            results += [' '.join(tokens[j:j+i])
                        for j in range(len(tokens) - i + 1)]
    return results


def get_sentiment(polarity):
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    return 0


def compute_feature_quality_score(sentiment, N=5):
    return 1. + (N - 1) / (1 + exp(-sentiment))


def chainer(s, sep=','):
    return list(chain.from_iterable(s.str.split(sep)))


def most_frequent(elements, counter, exclude=[]):
    element_freq = [counter[element] for element in elements if element not in exclude]
    if len(element_freq) == 0:
        return None
    return elements[element_freq.index(max(element_freq))]


def main(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    # load data
    print('Load data')
    df = pd.read_json(args.input, lines=True)
    print('Export ratings')
    ratings = df.loc[:, ['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    ratings.to_csv(
        os.path.join(args.out, 'rating.txt'), header=False, index=False)
    ratings['id'] = ratings['reviewerID'].map(
        str) + '-' + ratings['asin'].map(str)
    ratings = ratings.set_index('id')
    print('Export text')
    reviews = df[['reviewerID', 'asin', 'reviewText']]
    print('Lowercase all text')
    df['reviewText'] = df['reviewText'].str.lower()
    reviews.to_csv(os.path.join(
        args.out, 'review.txt'), sep='\t', header=False, index=False)
    print('Load concepts')
    concepts = pd.read_csv(args.concepts, header=None, sep='\t')
    concepts = list(set(concepts[0].tolist() + concepts[1].tolist()))
    print('Exclude stopwords')
    stop_words = set(stopwords.words('english'))
    concepts = [tok for tok in concepts if tok not in stop_words]
    all_text = ' '.join(df['reviewText'].tolist())
    tokens = [tok for tok in parse_line(all_text, n_grams=[1])]
    token_counter = Counter(tokens)
    concept_freq = pd.DataFrame(data={'concept': concepts})
    concept_freq['count'] = concept_freq['concept'].map(token_counter)
    print('Sort concept descending order')
    concept_freq = concept_freq.sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    print('Backup concepts count')
    concept_freq.to_csv(os.path.join(
        args.out, 'concept.cnt'), header=False, index=False)
    print('Export top %d concepts' % args.num_top_freq_aspect)
    top_freq_aspects = concept_freq[:args.num_top_freq_aspect]
    top_freq_aspects.to_csv(os.path.join(
        args.out, 'top_freq_aspects.txt'), header=False, index=False)
    top_freq_aspects = top_freq_aspects['concept'].tolist()
    # Exporting sentence sentiment
    with open(os.path.join(args.out, 'sentences.txt'), 'w') as f:
        for _, row in tqdm(df.iterrows(), desc='Exporting sentence sentiment'):
            review = TextBlob(row['reviewText'])
            for sentence in review.sentences:
                s = sentence.tokens
                if len(s) > 0 and s[-1] == '.':
                    s.remove('.')  # remove fullstop
                if len(s) > 0:
                    s = ' '.join(s)
                    f.write('{}\t{}\t{}\t{}\n'.format(
                        row['reviewerID'], row['asin'], s, get_sentiment(sentence.sentiment.polarity)))
    # Load sentences
    sentence_df = pd.read_csv(os.path.join(args.out, 'sentences.txt'), sep='\t', header=None,
                              names=['reviewerID', 'asin', 'sentence', 'sentiment'])
    # Exporting aspect sentiment
    with open(os.path.join(args.out, 'aspect_sentiment.txt'), 'w') as f:
        for user, item, sentence, sentiment in tqdm(zip(sentence_df['reviewerID'], sentence_df['asin'], sentence_df['sentence'], sentence_df['sentiment']), desc='Exporting aspect sentiment'):
            for tok in parse_line(str(sentence), n_grams=[1]):
                if tok in top_freq_aspects:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                        user, item, sentence, tok, sentiment))
    # Load aspect sentiment
    aspect_sentiments = pd.read_csv(os.path.join(args.out, 'aspect_sentiment.txt'), sep='\t',
                                    usecols=[0, 1, 3, 4], header=None,
                                    names=['reviewerID', 'asin', 'aspect', 'sentiment'])
    print('compute user, item aspect sentiment')
    aspect_sentiments = aspect_sentiments.groupby(
        by=['reviewerID', 'asin', 'aspect']).sum().reset_index()
    tqdm.pandas(desc='compute aspect score')
    aspect_sentiments['aspect_score'] = aspect_sentiments['sentiment'].progress_apply(
        lambda x: compute_feature_quality_score(x))
    print('export aspect scores to file')
    aspect_sentiments.to_csv(os.path.join(
        args.out, 'aspect_scores.csv'), index=False)
    aspect_sentiments['id'] = aspect_sentiments['reviewerID'].map(
        str) + '-' + aspect_sentiments['asin'].map(str)
    ratings['id'] = ratings['reviewerID'].map(
        str) + '-' + ratings['asin'].map(str)
    ratings = ratings.set_index('id')
    print('map ratings')
    aspect_sentiments['overall'] = aspect_sentiments['id'].map(
        ratings['overall'])
    print('compute correlation')
    aspect_sentiments = aspect_sentiments[aspect_sentiments['overall'].notnull(
    )]
    rating_aspect_correlations = aspect_sentiments.groupby(by=['aspect'])[['overall', 'aspect_score']].corr(
        method='pearson').iloc[0::2, -1].reset_index()[['aspect', 'aspect_score']].sort_values(by=['aspect_score'], ascending=False).reset_index(drop=True)
    print('export rating aspect correlation scores to file')
    rating_aspect_correlations.to_csv(os.path.join(
        args.out, 'rating_aspect_correlations.csv'), index=False, header=False)
    print('export top {} aspects to file'.format(args.num_top_corr_aspect))
    top_corr_aspects = rating_aspect_correlations[:args.num_top_corr_aspect]
    top_corr_aspects[['aspect']].to_csv(
        os.path.join(args.out, 'top_correlated_aspects.txt'), index=False, header=False)
    top_corr_aspects = set(top_corr_aspects['aspect'].tolist())
    # Load opinions
    opinions = set(pd.read_csv(args.positive_opinions, header=None)[
                   0].tolist() + pd.read_csv(args.negative_opinions, header=None)[0].tolist())
    with open(os.path.join(args.out, 'sentence_aspect_opinion_sentiments.txt'), 'w') as f:
        fieldnames = ['reviewerID', 'asin', 'sentence', 'aspect',
                      'aspect_count', 'opinion', 'opinion_count', 'sentiment']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for user, item, sentence, sentiment in tqdm(zip(sentence_df['reviewerID'], sentence_df['asin'], sentence_df['sentence'], sentence_df['sentiment']), total=len(sentence_df), desc='Exporting sentence aspect opinion'):
            sentence_aspects = set()
            sentence_opinions = set()
            for tok in parse_line(str(sentence).lower(), n_grams=[1]):
                if tok in opinions:
                    sentence_opinions.add(tok)
                elif tok in top_corr_aspects:
                    sentence_aspects.add(tok)
            if len(sentence_aspects) > 0 and len(sentence_opinions) > 0:
                writer.writerow({
                    'reviewerID': user,
                    'asin': item,
                    'sentence': sentence,
                    'aspect': '|'.join(list(sentence_aspects)),
                    'aspect_count': len(sentence_aspects),
                    'opinion': '|'.join(list(sentence_opinions)),
                    'opinion_count': len(sentence_opinions),
                    'sentiment': sentiment
                })
    # Export data
    data_df = pd.read_csv(os.path.join(args.out, 'sentence_aspect_opinion_sentiments.txt'))
    data_df = data_df[data_df['aspect'].notnull() & data_df['opinion'].notnull()]
    aspects = chainer(data_df['aspect'], '|')
    aspect_counter = Counter(aspects)
    opinions = chainer(data_df['opinion'], '|')
    opinion_counter = Counter(opinions)
    tqdm.pandas(desc='Spliting aspect')
    data_df['split_aspect'] = data_df['aspect'].progress_apply(lambda x: str(x).split('|'))
    tqdm.pandas(desc='Choosing most freq aspect')
    data_df['most_freq_aspect'] = data_df['split_aspect'].progress_apply(lambda x: most_frequent(x, aspect_counter))
    data_df['split_opinion'] = data_df['opinion'].apply(lambda x: str(x).split('|'))
    data_df['most_freq_opinion'] = data_df.apply(lambda row: most_frequent(row['split_opinion'], opinion_counter, row['most_freq_aspect'].split() + [row['most_freq_aspect']]), axis=1)
    print('Filter sentences without opinion')
    data_df = data_df[data_df['most_freq_opinion'].notnull()]
    data_df['aspect_term'] = data_df['most_freq_aspect'].apply(lambda x: '_'.join(str(x).split()))
    data_df['opinion_term'] = data_df['most_freq_opinion'].apply(lambda x: '_'.join(str(x).split()))
    data_df['sentence'] = data_df.apply(lambda row: ' '.join(word_tokenize(str(row['sentence']).replace(row['most_freq_aspect'], row['aspect_term']).replace(row['most_freq_opinion'], row['opinion_term']))), axis=1)
    data_df['aspect'] = data_df['aspect_term']
    data_df['opinion'] = data_df['opinion_term']
    tqdm.pandas(desc='Locating aspect position')
    data_df['aspect_pos'] = data_df.progress_apply(lambda row: word_tokenize(row['sentence']).index(str(row['aspect'])), axis=1)
    tqdm.pandas(desc='Locating opinion position')
    data_df['opinion_pos'] = data_df.progress_apply(lambda row: word_tokenize(row['sentence']).index(str(row['opinion'])), axis=1)
    tqdm.pandas(desc='Getting sentence length')
    data_df['sentence_len'] = data_df['sentence'].progress_apply(lambda x: len(str(x).split()))
    print('Export sentence with aspect and opinion positions to file')
    data_df[['reviewerID','asin','sentence','sentence_len','aspect','aspect_pos','opinion','opinion_pos']].to_csv(os.path.join(args.out, 'data.csv'), index=False)
    tqdm.pandas(desc='Grouping aspect opinion sentiment')
    data_df['aspect_opinion_sentiment'] = data_df.progress_apply(lambda row: '{}:{}:{}'.format(row['aspect'], row['opinion'], row['sentiment']), axis=1)
    print('Export efm sentiment file')
    efm_sentiment = data_df.groupby(['reviewerID', 'asin'])['aspect_opinion_sentiment'].apply(list).reset_index()
    efm_sentiment['aspect_opinion_sentiment'] = efm_sentiment['aspect_opinion_sentiment'].apply(lambda x: ','.join([str(i) for i in x]))
    with open(os.path.join(args.out, 'sentiment.txt'), 'w') as f:
        for _, row in tqdm(efm_sentiment.iterrows(), total=len(efm_sentiment), desc='Exporting sentiment data file'):
            f.write('{},{},{}\n'.format(row['reviewerID'], row['asin'], row['aspect_opinion_sentiment']))
    print('done')


if __name__ == '__main__':
    main(parse_arguments())

