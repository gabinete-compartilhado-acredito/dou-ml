
### Functions for machine learning ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import joblib
import os
import re
import json
import datetime
from zlib import crc32
from pip._vendor import pkg_resources
from scipy import stats
from multiprocessing import Pool, cpu_count
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit


###########################################
### Splitting datasets into random sets ###
###########################################

def shuffled_pos(length, seed):
    """
    Return indices from 0 to `length` - 1 in a shuffled state, given random `seed`.
    """
    return np.random.RandomState(seed=seed).permutation(length)


def random_index_sets(size, set_fracs, seed):
    """
    Return sets of random indices (from 0 to `size` - 1) with lengths 
    given by ~ `size` * `set_fracs`.
    
    
    Input
    -----
    
    size : int
        The size of the index list to split into sets.
        
    set_fracs : iterable
        The fractions of the list of indices that each index set 
        should contain. 
    
    seed : int
        The seed for the random number generator.
        
        
    Returns
    -------
    
    indices : tuple of arrays
        The indices for each set.
    """
    
    assert np.isclose(np.sum(set_fracs), 1), '`set_fracs` should add up to one.'
    
    # Create randomized list of indices:
    shuffled_indices = shuffled_pos(size, seed)
    
    
    indices   = []
    set_start = [0]
    # Determine the sizes of the sets:
    set_sizes = [round(size * f) for f in set_fracs]
    set_sizes[0] = size - sum(set_sizes[1:])
    assert np.sum(set_sizes) == size, 'Set sizes should add up to total size.'
    
    for i in range(0, len(set_fracs) - 1):
        # Select indices for a set:
        set_start.append(set_start[i] + set_sizes[i])
        set_indices = shuffled_indices[set_start[i]:set_start[i + 1]]
        indices.append(set_indices)
        assert len(indices[i]) == len(set(indices[i])), 'There are repeating indices in a set.'
        
    # Select the indices for the last set:
    indices.append(shuffled_indices[set_start[-1]:])
    assert len(set(np.concatenate(indices))) == sum([len(i) for i in indices]), \
    'There are common indices between sets.'
    
    return tuple(indices)


def random_set_split(df, set_fracs, seed):
    """
    Split a DataFrame into randomly selected disjoint and complete sets.
    
    
    Input
    -----
    
    df : Pandas DataFrame
        The dataframe to split into a complete and disjoint set of sub-sets.
        
    set_fracs : array-like
        The fraction of `df` that should be put into each set. The length of 
        `set_fracs` determines the number of sub-sets to create.
    
    seed : int
        The seed for the random number generator used to split `df`.
        
    
    Returns
    -------
    
    A tuple of DataFrames, one for each fraction in `set_fracs`, in that order.
    """
    # Get positional indices for each set:
    sets_idx = random_index_sets(len(df), set_fracs, seed)
    
    return tuple(df.iloc[idx] for idx in sets_idx)


def hash_string(string, prefix=''):
    """
    Takes a `string` as input, remove `prefix` from it and turns it into a hash.
    """
    name   = string.replace(prefix, '')
    return crc32(bytes(name, 'utf-8'))


def test_set_check_by_string(string, test_frac, prefix=''):
    """
    Returns a boolean array saying if the data identified by `string` belongs to the test set or not.
    
    Input
    -----
    
    string : str
        The string that uniquely identifies an example.
    
    test_frac : float
        The fraction of the complete dataset that should go to the test set (0 to 1).
        
    prefix : str (default '')
        A substring to remove from `string` before deciding where to place the example.
        
        
    Returns
    -------
    
    A bool number saying if the example belongs to the test set.
    """
    return hash_string(string, prefix) & 0xffffffff < test_frac * 2**32


def train_test_split_by_string(df, test_frac, col, prefix=''):
    """
    Split a DataFrame `df` into train and test sets based on string hashing.
    
    Input
    -----
    
    df : Pandas DataFrame
        The data to split.
        
    test_frac : float
        The fraction of `df` that should go to the test set (0 to 1).

    col : str or int
        The name of the `df` column to use as identifier (to be hashed).
        
    prefix : str (default '')
        A substring to remove from the rows in column `col` of `df` 
        before deciding where to place the example.
        
    Returns
    -------
    
    The train and the test sets (Pandas DataFrames).
    """
    ids = df[col]
    in_test_set = ids.apply(lambda s: test_set_check_by_string(s, test_frac, prefix))
    return df.loc[~in_test_set], df.loc[in_test_set]


def train_test_split_by_date(df, date_col, test_min_date):
    """
    Given a DataFrame `df` with a date column `date_col`, split it into 
    two disjoint DataFrames with data previous to date `test_min_date` 
    and data equal to or later than `test_min_date`. These are return 
    is this order.
    """
    train = df.loc[df[date_col] <  test_min_date]
    test  = df.loc[df[date_col] >= test_min_date]
    return train, test


def train_val_test_split_by_date_n_string(df, date_col, min_test_date, str_col, prefix=''):
    """
    Split a DataFrame `df` into 3 disjoint sets (train, validation and test):
    The first (train) contains all data with dates stored in column 
    `date_col` that are less than `min_test_date`. The last two sets (validation 
    and test) contain data from period equal to or later than `min_test_date`.
    These two sets have approximatelly the same size and are selected by 
    hashing strings in column `str_col`; these strings can have a substring
    `prefix` removed from them before hashing.    
    """
    # Quebra a base em passado (train) e futuro (test e validação):
    train, val_test = train_test_split_by_date(df, date_col, min_test_date)
    n_train    = len(train)
    n_val_test = len(val_test)

    # Quebra o futuro em test e validação usando string hashing:
    val, test = train_test_split_by_string(val_test, 0.5, str_col, prefix)
    n_val     = len(val)
    n_test    = len(test)

    print('# train:     ', n_train)
    print('# validation: {:d}  ({:.1f}%)'.format(n_val, n_val / (n_val + n_train) * 100))
    print('# test:       {:d}  ({:.1f}%)'.format(n_test, n_test / (n_test + n_train) * 100))   
    
    return train, val, test


def Xy_split(df, y_col):
    """
    Given a Pandas DataFrame `df` and a column name `y_col` (str), returns:
    - The features X (`df` without column `y_col`);
    - The target y (`df[y_col]`)  
    """
    out_df = df.copy()
    
    X = out_df.drop(y_col, axis=1)
    y = out_df[y_col]
    
    return X, y


def stratified_split(df, labels, test_size, random_state):
    """
    Return `df_train`, `df_test`, `labels_train`, `labels_test` by 
    splitting `df` (DataFrame) and `labels` (Series) into two sets 
    using Stratified sampling.
    """
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = tuple(strat_split.split(df, labels))[0]

    df_train = df.iloc[train_idx]
    df_test  = df.iloc[test_idx]
    
    labels_train = labels.iloc[train_idx]
    labels_test  = labels.iloc[test_idx]
    
    return df_train, df_test, labels_train, labels_test



######################
### New processors ###
######################


def log10_one_plus(x):
    """
    Return log10(1 + x) for `x` (array-like).
    """
    return np.log10(1 + x)


def sparse_max(A, B):
    """
    Return the element-wise maximum of sparse matrices `A` and `B`.
    """
    AgtB = (A > B).astype(int)
    M = AgtB.multiply(A - B) + B
    return M


def make_column_selector(column_list, default='passthrough', *, remainder='drop', sparse_threshold=0.3, 
                            n_jobs=None, transformer_weights=None, verbose=False):
    """
    Create a Column Transformer with one transformer of type `default` 
    per column listed in `column_list`. The name of each individual 
    transformer is the name of the respective column.

    The rest of the parameters are `ColumnTransformer` parameters.
    """
    
    if type(default) == str:
        transformers    = [(col, default, [col]) for col in column_list]
    else:
        transformers    = [(col, clone(default), [col]) for col in column_list]

    column_transformer  = ColumnTransformer(transformers, remainder, sparse_threshold, 
                                            n_jobs, transformer_weights, verbose)
    
    return column_transformer


class Predictor2Transformer(TransformerMixin):
    
    def __init__(self, predictor, transform_role='predict'):
        """
        A wrapper for sklearn Predictors that duck type them into
        Transformers. The resulting object behaves exactly like 
        the input Predictor, except that the Predictor method 
        specified by `transform_role` is renamed as a transform method.
        
        Parameters
        ----------
        
        predictor : sklearn Predictor
            An object with methods `fit` and at least one of the 
            following: `predict`, `predict_proba`, `predict_log_proba` 
            and `decision_function`.
            
        transfrom_role : str (default = 'predict')
            A string specifying which one of the Predictor 
            methods will be used when calling `transform`.
            It can be: "predict", "predict_proba", "predict_log_proba" 
            or "decision_function".
        """
        
        self.transform_role = transform_role
        self.predictor      = clone(predictor)
    
    def fit(self, X, y=None):
        self.predictor.fit(X, y)
        return self
    
    def transform(self, X):
        
        if self.transform_role == 'predict':
            return self.predictor.predict(X)
        
        elif self.transform_role == 'predict_proba':
            return self.predictor.predict_proba(X)
        
        elif self.transform_role == 'decision_function':
            return self.predictor.decision_function(X)
        
        elif self.transform_role == 'predict_log_proba':
            return self.predictor.predict_log_proba(X)
        
        else:
            raise Exception("Unknown '" + self.transform_role + "' `transform_role`.")
            
    def get_params(self, deep=True):
        
        params = self.predictor.get_params(deep)
        params['transform_role'] = self.transform_role
        
        return params
    
    def set_params(self, **params):
        
        if 'transform_role' in params.keys():
            self.transform_role = params['transform_role']
            params.pop('transform_role')
            
        self.predictor.set_params(**params)
        

class WeightedVectorizer(CountVectorizer):
    """Similar to CountVectorizer, but multiplies the token counts 
    by `keywords_weight` if they refer to tokens created by applying 
    the vectorizer to `keywords` and not to `anti_keywords`.
    Parameters
    ----------
    input : string {'filename', 'file', 'content'}, default='content'
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be a sequence of items that
        can be of type string or byte.
    encoding : string, default='utf-8'
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode'}, default=None
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
    preprocessor : callable, default=None
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
        Only applies if ``analyzer is not callable``.
    tokenizer : callable, default=None
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    stop_words : string {'english'}, list, default=None
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    ngram_range : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.
    analyzer : string, {'word', 'char', 'char_wb'} or callable, \
            default='word'
        Whether the feature should be made of word n-gram or character
        n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
        .. versionchanged:: 0.21
        Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
        first read from the file and then passed to the given callable
        analyzer.
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    binary : bool, default=False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().
    keywords : list of str
        Texts to be transformed into tokens that should be weighted by 
        `keywords_weight`. Each entry may contain more than one word.
    anti_keywords : list of str
        Texts to be transformed into tokens that should NOT be weighted by 
        `keywords_weight`, even if they are also produced from `keywords`.
        This is useful to avoid only keep 2-grams when `ngram_range=(1,2)`, 
        for instance.
    keywords_weight : int or float
        Weight to be applied to counts of tokens created according to `keywords` 
        and `anti_keywords`.
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    fixed_vocabulary_: boolean
        True if a fixed vocabulary of term to indices mapping
        is provided by the user
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    
  
    """
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None, stop_words=None,
                 token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False, dtype=np.int64, keywords=None, 
                 anti_keywords=None, keywords_weight=1):
        
        # Initialize Ordinary CountVectorizer:
        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, 
                         preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern, 
                         ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features, 
                         vocabulary=vocabulary, binary=binary, dtype=dtype)
        
        # Get extra parameters:
        self.keywords = keywords
        self.anti_keywords = anti_keywords
        self.keywords_weight = keywords_weight
        
        # Check for consistency:
        if (type(keywords_weight) == float or type(keywords_weight) == np.float64) and \
           (dtype != np.float64 and dtype != float):
            raise Exception("To use float-like `keywords_weight`, change `dtype` to `float` or `np.float64`.")
    
    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self
    
    def transform(self, raw_documents):

        # Vectorize documents:
        X = super().transform(raw_documents)
        # Multiply keyword tokens by weights:
        if self.keywords != None:
            X[:, self.weighted_tokens_idx_] = X[:, self.weighted_tokens_idx_] * self.keywords_weight 
        
        return X
    
    def fit_transform(self, raw_documents, y=None):
        
        ### Fit ###
        if self.keywords != None: 
            # Fit vectorizer to keywords:
            super().fit_transform(self.keywords)
            processed_keywords = self.get_feature_names()
            
            # If no anti-keywords were provided, these are the keyword tokens:
            if self.anti_keywords == None:
                key_tokens = processed_keywords   
            else:
                # Fit vectorizer to anti-keywords:
                super().fit_transform(self.anti_keywords)
                processed_anti_keywords = self.get_feature_names()
                # Remove anti-keywords from token list:
                key_tokens = list(set(processed_keywords) - set(processed_anti_keywords))
        
        ### Final fit and transform ###
        
        # Fit CountVectorizer to actual data:
        X = super().fit_transform(raw_documents, y)
        
        # Get index of tokens to be weighted.
        if self.keywords != None:
            model_tokens = self.get_feature_names()
            self.weighted_tokens_ = list(set(key_tokens) & set(model_tokens))
            self.weighted_tokens_idx_ = [self.vocabulary_[token] for token in self.weighted_tokens_]
            # Multiply keyword tokens by weights:
            X[:, self.weighted_tokens_idx_] = X[:, self.weighted_tokens_idx_] * self.keywords_weight 
        
        return X
    

class MultiColumnVectorizer(BaseEstimator, TransformerMixin):
    """
    Vectorize text columns into a single term frequency matrix, giving 
    different weights to text that appears on different columns. It creates
    a single vocabulary from all columns, transforms each column separately 
    and then aggregate them. 
    
    Input
    -----
            
    text_cols : list of str
        List of `input_df` column names that contain text.
        
    text_weights : list of float
        Weights that multiply the counts made by `vectorizer` on 
        each one of `text_cols` columns.
        
    agg : str
        How to aggregate the counts from each column:
        - 'sum': Sum the counts from each column.
        - 'max': Get the maximum of the counts from the columns.
    
    kwargs : keyword arguments for the sklearn CountVectorizer object
        Parameters for the vectorizer that will be used to vectorize the text columns.
        
    input_df : Pandas DataFrame
        Dataframe containing text columns to be vectorized.

    
    Transform Return
    ----------------
    
    agg_counts : numpy array or sparse matrix
        The aggregated counts of the vocabulary from each column.
    """
    def __init__(self, text_cols, text_weights, agg='max',
                 input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
    
        # Load parameters:
        self.text_cols    = text_cols
        self.text_weights = text_weights
        self.agg          = agg
        self.input        = input
        self.encoding     = encoding
        self.decode_error = decode_error
        self.strip_accents= strip_accents
        self.lowercase    = lowercase
        self.preprocessor = preprocessor
        self.tokenizer    = tokenizer
        self.stop_words   = stop_words
        self.token_pattern= token_pattern
        self.ngram_range  = ngram_range
        self.analyzer     = analyzer
        self.max_df       = max_df
        self.min_df       = min_df
        self.max_features = max_features
        self.vocabulary   = vocabulary
        self.binary       = binary
        self.dtype        = dtype
        self.vec          = CountVectorizer(binary=self.binary, lowercase=self.lowercase, max_df=self.max_df, min_df=self.min_df, max_features=self.max_features,
                                            stop_words=self.stop_words, vocabulary=self.vocabulary, ngram_range=self.ngram_range)
        #input=self.input, encoding=self.encoding, decode_error=self.decode_error, strip_accents=self.strip_accents,
        #lowercase=self.lowercase, preprocessor=self.preprocessor, tokenizer=self.tokenizer,
        #stop_words=self.stop_words, token_pattern=self.token_pattern, ngram_range=self.ngram_range,
        #analyzer=self.analyzer, max_df=self.max_df, min_df=self.min_df, max_features=self.max_features,
        #vocabulary=self.vocabulary, binary=self.binary, dtype=self.dtype)
        
        # Valid input checks:
        if self.agg != 'sum' and self.agg != 'max':
            raise Exception("Unknown `agg = " + str(self.agg) + "` option. It should be 'sum' or 'max'.") 
        assert len(self.text_cols) == len(self.text_weights), '`text_cols` and `text_weights` must have the same length'
    
    def fit(self, X, y=None):
        
        # Select text columns and fill missing values with empty strings:
        text_df   = X[self.text_cols].fillna('')       
        # Create a single column with all texts:
        all_text  = text_df.agg(' '.join, axis=1)
        
        # Fit vectorizer to all text:
        self.vec.fit(all_text)
        
        return self
    
    def transform(self, X):
         
        # Select text columns and fill missing values with empty strings:
        text_df   = X[self.text_cols].fillna('')       
       
        # Vectorize all text columns:
        count_matrices = np.array([w * self.vec.transform(text_df[col]) \
                                       for w, col in zip(self.text_weights, self.text_cols)])
        # Aggregate the results:
        
        # Keep the largest weight:
        if self.agg == 'max':
            agg_counts = count_matrices[0]
            if len(count_matrices) > 1:
                for count_matrix in count_matrices[1:]:
                    agg_counts = sparse_max(agg_counts, count_matrix)
        # Sum the weights:
        elif self.agg == 'sum':
            count_matrices = np.array([w * self.vec.transform(text_df[col]) \
                                       for w, col in zip(self.text_weights, self.text_cols)])
            agg_counts = count_matrices.sum(axis=0)
        else:
            raise Exception("Unknown `agg = " + str(self.agg) + "` option. It should be 'sum' or 'max'.") 

        return agg_counts


########################
### Estimating tools ###
########################


def decision_tree_balanced_weights(y):
    """
    Given an array of labels `y`, return the value of the 
    weights (array) for each class that corresponds to the 
    'balanced' weights in a decision tree. Note that the 
    weights are normalized so class 0 has weight 1.
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    raw_weights = n_samples / (n_classes * np.bincount(y))
    
    return raw_weights / raw_weights[0]


def adjusted_r2(linear_regression, X, y, w=None):
    """
    Compute the adjusted R^2 coefficient for 
    the `linear_regression` (sklearn linear 
    regression model) w.r.t. measurements 
    `X` and targets `y`. `w` are the sample
    weights.
    """
    
    n  = X.shape[0]
    p  = X.shape[1]
    r2 = linear_regression.score(X, y, w)
    
    adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return adjusted


##################
### New models ###
##################


class MultiTypeNB(BaseEstimator):
    
    def __init__(self, p1, p2, decision_threshold=None):
        """
        Given two Pipelines `p1` and `p2` that end on a Naive Bayes 
        predictor, combine their predictions into one by multiplying 
        their probabilities and renormalizing.

        Both pipelines act on all columns of X, so they should 
        contain a column selection operation.

        Parameters
        ----------

        decision_threshold : float (default None)
            Probability threshold when running `predict` method.
            If None, return the class with the highest probability.
            If float, assume that the classification is binary 
            and predict 1 if the second class probability is above
            `decision_threshold`.
        """
        self.p1 = p1
        self.p2 = p2
        self.decision_threshold = decision_threshold
        
    def fit(self, X, y):
        self.p1.fit(X, y)
        self.p2.fit(X, y)
        
    def predict_proba(self, X):
        
        # Compute log probability for individual predictors:
        cat_log_proba  = self.p1.predict_log_proba(X)
        cont_log_proba = self.p2.predict_log_proba(X)
        
        # Combine both predictions:
        prior = self.p2.steps[-1][1].class_prior_
        proba = np.exp(cat_log_proba + cont_log_proba - np.log(prior))
        proba = proba / proba.sum(axis=1)[:, None]
        
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        if self.decision_threshold == None:
            return np.argmax(proba, axis=1)
        else:
            return (proba[:, 1] >= self.decision_threshold).astype(int)

class ConstantPicker(BaseEstimator):
    """
    A model that predict always the same value for every example.

    Input
    -----

    constant_value : int or float
        The value that will be output as prediction for all the examples.
    """    
    def __init__(self, constant_value):
        self.constant_value = constant_value
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return self.constant_value * np.ones(X.shape[0])


class RandomPicker(BaseEstimator):
    """
    A model whose predictions are random.

    Input
    -----

    follow_frequency : bool (default False)
        If True, randomly samples from the y sample used to train,
        so the frequency of each value follows the training sample.
        If False, sample each value uniformly.
    """    
    def __init__(self, follow_frequency=False):
        self.follow_frequency = follow_frequency
    
    def fit(self, X, y=None):
        self.y_sample = y
        return self
    
    def predict(self, X):
        n = X.shape[0]
        
        if self.follow_frequency:
            choices = self.y_sample
            return np.random.choice(choices, n)
        
        else:
            choices = np.unique(self.y_sample)
            return np.random.choice(choices, n)


class UnsupervisedEnsemble:
    """
    A Bagging model for unsupervised predictors such as OneClassSVM. 
    It fits multiple clones of the base estimator to bootstrapped 
    (resampling with replacement) samples of the data. When predicting,
    it uses hard-voting (majority wins).
    
    base_estimator : predictor
        An object that implements fit and predict methods, the individual 
        predictor that will classify the data.
    
    max_samples : int
        Maximum number of samples to include in the bootstrapped samples.
        These subsamples will have a size of the input data or `max_samples`, 
        whatever is lower.
    
    n_estimators : int
        Number of copies of the base estimator that will be used in the 
        voting ensemple. Each one is trained on a bootstrapped sample.
        
    n_jobs : int
        Number of jobs for parallel processing in fitting and predicting.
        
    random_state : int
        The seed for the random number generator. Currently, the 
        subsample used in each estimator is generated from a seed
        that is `random_state` + [0, 1, ..., `n_estimators` - 1].
    """
    
    def __init__(self, base_estimator, max_samples, n_estimators=4, n_jobs=2, random_state=None):
        
        # Get input parameters:
        self.n_estimators   = n_estimators
        self.max_samples    = max_samples
        self.random_state   = random_state
        self.base_estimator = base_estimator
        self.n_jobs         = n_jobs

        
    def fit(self, X):
        
        # Compute subsample sizes:
        n_samples = len(X)
        subsample_size = np.min([n_samples, self.max_samples])
        
        # Function that fits the estimator to a bootstrapped sample:
        def thread(X, subsample_size, estimator, seed):
            X_subsample = resample(X, n_samples=subsample_size, random_state=seed)
            m = clone(estimator)
            m.fit(X_subsample)
            return m
            
        # Parallel Loop over fitting estimators:
        seeds = self.random_state + np.arange(self.n_estimators)
        p = joblib.Parallel(n_jobs=self.n_jobs)
        self.estimators = p(joblib.delayed(thread)(X, subsample_size, self.base_estimator, seed) for seed in seeds)
        
    def predict(self, X):
        
        # Function for predicting:
        def thread(estimator, X):
            return estimator.predict(X)
        
        # Predict each estimator:
        p = joblib.Parallel(n_jobs=self.n_jobs)
        y = p(joblib.delayed(thread)(estimator, X) for estimator in self.estimators)

        # Get majority of votes:
        y = stats.mode(y, axis=0)[0][0]
        
        return y

    def fit_predict(self, X):

        self.fit(X)
        y = self.predict(X)
        return y
    
    
#########################
### Evaluating models ###
#########################


def mean_dev_scores(scores): 
    """
    Given a numpy array `scores` return its mean and
    standard deviation as a tuple.
    """
    mean_score = np.array(scores).mean()
    dev_score  = np.array(scores).std()
    
    return mean_score, dev_score


def baseline_random(X, y, scoring, cv=5, best_guess=None, verbose=False):
    """
    Computes score for random (dumb) predictors. Currently the implementation
    only works for classification (for regression, guessing the mean value 
    makes sense). The best average score of the following predictors is returned:
    -- Constant         (always returns the same value);
    -- Random Uniform   (returns a random target with equal probability);
    -- Follow Frequency (returns a random target following their frequency 
                         in `y`).
    
    Input
    -----
    
    X : array-like
        Matrix of features (actually it is not used, can be anything).
    
    y : array-like
        Target values. The random choices model use this to sample a 
        random value, either uniformly or according to its frequency 
        in `y`.
        
    best_guess : int or float (default None)
        The value to return for the "Constant" model. If `None`, 
        use the mode of `y` if `y` contains integers or the mean, 
        otherwise.
    
    scoring : str or sklearn scorer
        The scorer to use to evaluate the random choice model.
        
    cv : int (default 5)
        Number of cross-validation sets used to estimate the performance.
    
    verbose : bool (default False)
        Whether information about the performance of all the random 
        models are printed out.
        
    
    Returns
    -------
    
    mean_max : float
        The average score of the best performing random model
        (in terms of average score).
    
    dev_max : float
        The std. dev. of the score of the best performing model
        (in terms of average score).
    """
    
    # Define
    if best_guess == None:
        if y.dtype.kind == 'i':
            best_guess, dump = stats.mode(y)
            best_guess = best_guess[0]
        elif y.dtype.kind == 'f':
            best_guess = np.mean(y)
        else:
            raise Exception('y has unknkown data type.')
    if verbose:
        print('Constant value to be used:', best_guess)
    
    # Instantiate random models:
    constant_picker    = ConstantPicker(best_guess)
    random_picker      = RandomPicker()
    random_picker_freq = RandomPicker(follow_frequency=True)
    # Create list of models:
    model_list = [('Constant', constant_picker), ('Random Uniform', random_picker), 
                  ('Follow frequency', random_picker_freq)]

    # Loop over mode
    name_max = None; mean_max = None; dev_max = None
    for name_model in model_list:
        name  = name_model[0]
        model = name_model[1]
        if verbose:
            print(model)
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        mean, dev = mean_dev_scores(scores)
        if verbose:
            print('Mean and dev:', mean, dev)
        if mean_max == None or mean > mean_max:
            mean_max = mean
            dev_max  = dev
            name_max = name
    if verbose:
        print('Best random model is:', name_max)
    return mean_max, dev_max


def get_entries(array_like, idx):
    """
    Given a array-like set `array_like` and a array-like list of 
    positions `idx`, returns the subset of `array_like` formed 
    by its elements in positions `idx`.
    """
    if type(array_like) == np.ndarray:
        subset = array_like[idx]
    else: # Assume pandas object:
        subset = array_like.iloc[idx]
    return subset


def cv_regression_score(pipeline, X, y, cv=5, shuffle_seed=-1, n_jobs=None):
    """
    Compute the average RMSE and its std. dev. of a regression using cross-validation
    
    
    Input
    -----
    
    pipeline : a sklearn predictor
        The model used to predict y from X.
    
    X : Pandas DataFrame or numpy array
        The features (independent variables) used to predict y. Note that 
        its data type is defined by what `pipeline` expects.
    
    y : array-like
        The target (dependent) variable to be predicted.
        
    cv : int (default 5)
        Ǹumber of cross-validation sets.
    
    shuffle_seed : int or Nonetype (default -1)
        Seed for random number generator used to shuffle the dataset 
        before splitting into train and test sets. Set to -1 if
        no shuffling is required, and to None to get a random seed.
    
    n_jobs : int or Nonetype (default None)
        Number of parallel processes used to run the cross-validation.
        Check sklearn.model_selection.cross_val_score docstring.
    
    
    Returns
    -------
    
    mean_score : float
        The average Root Mean Squared Error (RMSE) of the cross-validation steps.
        
    dev_score : float
        The standard deviation of the RMSEs obtained in cross-validation.
    """
    
    assert len(y) == len(X), 'X and y does not have the same number of entries'
    
    # Shuffle data if requested:
    if shuffle_seed != -1:
        random_order = np.random.RandomState(seed=shuffle_seed).permutation(len(y)) 
        # Shuffle X and y:
        Xs = get_entries(X, random_order)      
        ys = get_entries(y, random_order)

    else:
        Xs = X
        ys = y
    
    # Compute scores in cross- validation:
    neg_scores2 = cross_val_score(pipeline, Xs, ys, scoring='neg_mean_squared_error', cv=cv, n_jobs=n_jobs)
    scores = np.sqrt(-neg_scores2)
    
    # Compute statistics:
    mean_score = scores.mean()
    dev_score  = scores.std()
    
    return mean_score, dev_score


def train_regression_score(pipeline, X, y, cv=5, shuffle_seed=-1, n_jobs=None):
    """
    Compute the average RMSE and its std. dev. of a regression using 
    the training sample. It still split the data into subsets to use 
    a similar data size to a validation process.
    
    
    Input
    -----
    
    pipeline : a sklearn predictor
        The model used to predict y from X.
    
    X : Pandas DataFrame or numpy array
        The features (independent variables) used to predict y. Note that 
        its data type is defined by what `pipeline` expects.
    
    y : array-like
        The target (dependent) variable to be predicted.
        
    cv : int (default 5)
        Ǹumber of cross-validation sets.
    
    shuffle_seed : int or Nonetype (default -1)
        Seed for random number generator used to shuffle the dataset 
        before splitting into train and test sets. Set to -1 if
        no shuffling is required, and to None to get a random seed.
    
    n_jobs : int or Nonetype (default None)
        Number of parallel processes used to run the cross-validation.
        Check sklearn.model_selection.cross_val_score docstring.
    
    
    Returns
    -------
    
    mean_score : float
        The average Root Mean Squared Error (RMSE) of the cross-validation steps.
        
    dev_score : float
        The standard deviation of the RMSEs obtained in cross-validation.
    """
    
    assert len(y) == len(X), 'X and y does not have the same number of entries'
    
    # Shuffle data if requested:
    if shuffle_seed != -1:
        random_order = np.random.RandomState(seed=shuffle_seed).permutation(len(y)) 
        # Shuffle X and y:
        Xs = get_entries(X, random_order)      
        ys = get_entries(y, random_order)

    else:
        Xs = X
        ys = y
        
    # Prepare parallel processing:
    if n_jobs == -1:
        n_jobs = cpu_count()
    if n_jobs == None:
        n_jobs = 1
    pool = Pool(processes=n_jobs)
    
    # Prepare k-folds:
    kf = KFold(n_splits=cv)
    kf_train_indices = [a[0] for a in list(kf.split(Xs, ys))]
    
    def compute_scores(train_index):
        # Get train subset:
        kf_X_train = get_entries(Xs, train_index)
        kf_y_train = get_entries(ys, train_index)
        # Fit model and predict:
        pipeline.fit(kf_X_train, kf_y_train)
        kf_y_pred  = pipeline.predict(kf_X_train)
        # Compute the score:
        score = np.sqrt(mean_squared_error(kf_y_train, kf_y_pred))
        return score   
    
    # Run k-folds:
    scores = [compute_scores(train_index) for train_index in kf_train_indices]
    #scores = pool.map(compute_scores, kf_train_indices)
    
    # Compute statistics:
    mean_score = np.array(scores).mean()
    dev_score  = np.array(scores).std()
    
    return mean_score, dev_score


def rand_regression_score(pipeline, X, y, cv=5, rand_y_seed=None, shuffle_seed=-1, n_jobs=None):
    """
    Compute the average RMSE and its std. dev. of a regression for y 
    randomly assigned to X (fit noise).
    
    
    Input
    -----
    
    pipeline : a sklearn predictor
        The model used to predict y from X.
    
    X : Pandas DataFrame or numpy array
        The features (independent variables) used to predict y. Note that 
        its data type is defined by what `pipeline` expects.
    
    y : array-like
        The target (dependent) variable to be predicted.
        
    cv : int (default 5)
        Ǹumber of cross-validation sets.
    
    rand_y_seed : int or Nonetype (default None)
        Seed for the random number generator used to shuffle y alone
        (and erase its relation with X).
    
    shuffle_seed : int or Nonetype (default -1)
        Seed for random number generator used to shuffle the dataset 
        before splitting into train and test sets. Set to -1 if
        no shuffling is required, and to None to get a random seed.
    
    n_jobs : int or Nonetype (default None)
        Number of parallel processes used to run the cross-validation.
        Check sklearn.model_selection.cross_val_score docstring.
    
    
    Returns
    -------
    
    mean_score : float
        The average Root Mean Squared Error (RMSE) of the cross-validation steps.
        
    dev_score : float
        The standard deviation of the RMSEs obtained in cross-validation.
    """
    
    assert len(y) == len(X), 'X and y does not have the same number of entries'
    
    # Shuffle data if requested:
    if shuffle_seed != -1:
        random_order = np.random.RandomState(seed=shuffle_seed).permutation(len(y)) 
        # Shuffle X and y:
        Xs = get_entries(X, random_order)      
        ys = get_entries(y, random_order)

    else:
        Xs = X
        ys = y
    
    # Shuffle only y:
    random_y = np.random.RandomState(seed=rand_y_seed).permutation(len(y))
    ys = get_entries(ys, random_y)
    
    # Compute scores in cross- validation:
    neg_scores2 = cross_val_score(pipeline, Xs, ys, scoring='neg_mean_squared_error', cv=cv, n_jobs=n_jobs)
    scores = np.sqrt(-neg_scores2)
    
    # Compute statistics:
    mean_score = scores.mean()
    dev_score  = scores.std()
    
    return mean_score, dev_score


#####################################
### Plots and tables for analysis ###
#####################################


def searchCV_table(grid, sort_score='score'):
    """
    Given a scikit-learn `RandomizedSearchCV` result `grid`, return a 
    Pandas DataFrame sorted by descending mean test score with the following
    columns:
    -- Mean fit time;
    -- Std. dev. of fit time;
    -- Mean test score;
    -- Std. dev. of test score;
    -- All model parameters used by `RandomizedSearchCV`.
    """
    grid_df = pd.DataFrame(grid.cv_results_)
    #param_list = list(filter(lambda s: s[:6] == 'param_', grid_df.columns))
    #grid_cols = ['mean_fit_time', 'std_fit_time', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score'] + param_list
    col_regex = '_fit_time|mean_train_|std_train_|mean_test_|std_test_|param_'
    grid_cols = list(filter(lambda s: re.search(col_regex, s) != None, grid_df.columns)) 

    table = grid_df.sort_values('mean_test_' + sort_score, ascending=False)[grid_cols]
    return table


def plot_pr_roc_curves(classifier, X, y):
    """
    Plot the Precision and Recall relations to decision function 
    threshold and the Receiver Operating Characteristic (ROC) 
    curve. 
    
    I guess it only works for binary classification.
    
    Input
    -----
    
    classifier : sklearn predictor
        The classification model that will be used for prediction.
    
    X : Pandas DataFrame or array-like
        The matrix of features.
        
    y : array-like
        The true labels.
    """
    # Get decision function values for each example (sometimes called score):
    decisions = cross_val_predict(classifier, X, y, cv=3, method='decision_function')

    # Compute PR curve:
    precision, recall, threshold = precision_recall_curve(y, decisions)
    # Compute ROC curve:
    false_pos, true_pos, roc_thres = roc_curve(y, decisions)
    default_thres_pos = np.argmin(np.abs(roc_thres))

    pl.figure(figsize=(12,4))

    # PR plot:
    pl.subplot(1,2,1)
    pl.plot(threshold, precision[:-1], 'r-', label='Precision')
    pl.plot(threshold, recall[:-1], 'b--', label='Recall')
    pl.legend(fontsize=12)
    pl.tick_params(labelsize=14)
    pl.xlabel('Threshold (decision function)', fontsize=14)
    pl.grid(color='lightgray')

    # ROC plot:
    pl.subplot(1,2,2)
    pl.plot(false_pos, true_pos, 'g-', label='Model')
    pl.plot([0,1], [0,1], 'k--', label='Random')
    pl.scatter([false_pos[default_thres_pos]], [true_pos[default_thres_pos]], marker='o', 
            color='darkgreen', label='Default threshold')
    pl.legend(fontsize=12, loc='lower right')
    pl.tick_params(labelsize=14)
    pl.xlabel('False positive rate', fontsize=14)
    pl.ylabel('True positive rate', fontsize=14)
    pl.grid(color='lightgray')


def plot_learning_curve(model, X, y, scorer, shuffle=True, train_sizes=np.arange(0.10,1.10,0.10), random=False):
    """
    Plot the learning curve (score vs. training set size) for the training and test sets.

    
    Input
    -----

    model : sklearn predictor
        Model used to predict the target variables.

    X : array-like, shape (n_samples, n_features)
        Matrix of features, where n_samples is the number of examples and n_features is 
        the number of features.

    y : array-like, shape (n_samples)
        Target relative to X for classification or regression; None for unsupervised 
        learning.

    scorer : string, callable or None
        Metric used to measure the performance of the `model` (e.g. 'accuracy', 'f1', 
        'mean_squared_error').

    train_sizes : array-like (default 10% increment from 10% to 100% of total dataset size)
        Relative or absolute numbers of training examples that will be used to generate the 
        learning curve. If the dtype is float, it is regarded as a fraction of the maximum 
        size of the training set (that is determined by the selected validation method), 
        i.e. it has to be within (0, 1]. Otherwise it is interpreted as absolute sizes of 
        the training sets. Note that for classification the number of samples usually have 
        to be big enough to contain at least one sample from each class. 


    Output
    ------

    A matplotlib plot of the learning curve for the train and test sets.
    """
    # Compute learning curve:
    train_size, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, scoring=scorer, 
                                                           shuffle=shuffle)
    # Get averages and deviations:
    train_score_means = train_scores.mean(axis=1)
    train_score_devs  = train_scores.std(axis=1)
    test_score_means  = test_scores.mean(axis=1)
    test_score_devs   = test_scores.std(axis=1)

    if random:
        mean_random, dev_random = baseline_random(X, y, scorer)
        pl.fill_between(train_size, mean_random - dev_random, mean_random + dev_random, color='lightgray')
        pl.plot(train_size, mean_random * np.ones(len(train_size)), 'k--', label='Random')
    
    # Plot:
    pl.errorbar(train_size, train_score_means, yerr=train_score_devs, color='r', marker='.', label='Training set')
    pl.errorbar(train_size, test_score_means, yerr=test_score_devs, color='b', marker='.', label='Validation set')
    # Format:
    pl.xlabel('Training sample size', fontsize=14)
    pl.ylabel(scorer, fontsize=14)
    pl.tick_params(labelsize=14)
    pl.legend(fontsize=12)
    

def is_pandas_Q(x):
    """
    Returs True if `x` is a Pandas Series or Pandas DataFrame, and False otherwise. 
    """
    if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
        return True
    else:
        return False
    

def compute_learning_curve(model, metric, X_train, y_train, X_val, y_val, steps, 
                           shuffle=True, seed=None, verbose=False):
    """
    Compute the learning curve of a model, that is, the evolution of a 
    metric (currently accuracy) as a function of training sample size.
    The training sample is incremented with more examples in the order 
    they appear in the training set (check the `shuffle` parameter though).
    
    
    Input
    -----
    
    model : sklearn Pipeline or analogous.
        The machine learning model to train and score.
        
    metric : function
        A function that takes two arguments, y_true and y_predicted,
        and returns a float value (e.g. accuracy, RMS error, F1 score).

    X_train : Pandas DataFrame, Series or numpy array.
        The independent features used to train the `model`. Each row is 
        an example, and each column, a feature.
        
    y_train : Pandas Series or numpy array.
        The dependent variable that the `model` use as training labels/targets.
        It should be aligned with `X_train`.
    
    X_val : Pandas DataFrame, Series or numpy array.
        The validation independent features, used to test the `model`. It 
        assumes the same structure as `X_train`.
        
    y_val : Pandas Series or numpy array.
        The validation dependent variable, used to test the `model`. It 
        assumes the same structure as `y_train`.
        
    steps : The number of steps to take from the zero sample size to the 
        full `X_train` and `y_train` sizes. The first one is first 
        non-zero size if the interval from zero to full size was divided by 
        `steps`.
        
    shuffle : bool (default True)
        Whether or not to shuffle `X_train` and `y_train` (maintaining 
        alignment) before selecting into different sample sizes. 
        
    seed : int or None (default None)
        The seed used by the random shuffling. Set to None to use a random
        seed.
        
    verbose : bool (default False)
        Whether or not to print the training sample sizes.
        
    
    Returns
    -------
    
    sample_size : list of int
        The sizes of the training samples used to train and test the `model`.
        
    acc_train_vs_size : list of float
        The scores obtained when evaluating the `model` (trained with the 
        training set of size given by `sample_size`) with this same set.
    
    acc_val_vs_size : list of float
        The scores obtained when evaluating the `model` (trained with the 
        training set of size given by `sample_size`) with the full test set 
        `X_val` and `y_val`.
    """
    
    # Sanity checks:
    assert len(X_train) == len(y_train), '`X_train` and `y_train` must have the size length.'
    assert len(X_val)   == len(y_val),   '`X_val` and `y_val` must have the size length.'
    
    # Get training sample sizes:
    max_size = len(y_train)
    delta    = int(max_size / steps)
    sample_size = list(range(delta, max_size + 1, delta))
 
    # Shuffle the training set if requested:
    if shuffle:
        shuffled = shuffled_pos(len(y_train), seed)
        
        # If X_train is Pandas thing:
        if is_pandas_Q(X_train):
            X_t = X_train.iloc[shuffled]
        else:
            X_t = X_train[shuffled]
        
        # If y_train is Pandas thing:
        if is_pandas_Q(y_train):
            y_t = y_train.iloc[shuffled]
        else:
            y_t = y_train[shuffled]

    # No shuffling requested:
    else:
        X_t = X_train
        y_t = y_train
     
    # Loop over training sample sizes:
    acc_train_vs_size = []
    acc_val_vs_size   = []
    for n in sample_size:
        if verbose:
            print('Sample size', n)

        # Get training sample part:
        if is_pandas_Q(X_t):
            train_sample = X_t.iloc[:n]
        else:
            train_sample = X_t[:n]
        if is_pandas_Q(y_t):
            y_train_sample = y_t.iloc[:n]
        else:
            y_train_sample = y_t[:n]

        # Fit model:
        dump = model.fit(train_sample, y_train_sample)

        # Evaluate accuracy on training set:
        p_train = model.predict(train_sample)
        acc_train_vs_size.append(metric(y_train_sample, p_train))

        # Evaluate accuracy on validation set:
        p_val = model.predict(X_val)
        acc_val_vs_size.append(metric(y_val, p_val))

    return sample_size, acc_train_vs_size, acc_val_vs_size


def plot_learning_curve_old(model, metric, X_train, y_train, X_val, y_val, steps, shuffle=True, seed=None, verbose=False):
    """
    Plot the learning curve of a model, that is, the evolution of a 
    metric as a function of training sample size, both for the training 
    set and the validation set.
    
    
    Input
    -----
    
    model : sklearn Pipeline or analogous.
        The machine learning model to train and score.

    metric : function
        A function that takes two arguments, y_true and y_predicted,
        and returns a float value (e.g. accuracy, RMS error, F1 score).
        
    X_train : Pandas DataFrame, Series or numpy array.
        The independent features used to train the `model`. Each row is 
        an example, and each column, a feature.
        
    y_train : Pandas Series or numpy array.
        The dependent variable that the `model` use as training labels/targets.
        It should be aligned with `X_train`.
    
    X_val : Pandas DataFrame, Series or numpy array.
        The validation independent features, used to test the `model`. It 
        assumes the same structure as `X_train`.
        
    y_val : Pandas Series or numpy array.
        The validation dependent variable, used to test the `model`. It 
        assumes the same structure as `y_train`.
        
    steps : The number of steps to take from the zero sample size to the 
        full `X_train` and `y_train` sizes. The first one is first 
        non-zero size if the interval from zero to full size was divided by 
        `steps`.
        
    shuffle : bool (default True)
        Whether or not to shuffle `X_train` and `y_train` (maintaining 
        alignment) before selecting into different sample sizes. 
        
    seed : int or None (default None)
        The seed used by the random shuffling. Set to None to use a random
        seed.
        
    verbose : bool (default False)
        Whether or not to print the training sample sizes.
    """
    
    # Compute learning curve:
    sample_size, acc_train_vs_size, acc_val_vs_size = compute_learning_curve(model, metric, X_train, y_train, X_val, y_val, steps, shuffle, seed, verbose)
    
    pl.plot(sample_size, acc_train_vs_size, 'r-', marker='.', label='Training set')
    pl.plot(sample_size, acc_val_vs_size,   'b-', marker='.', label='Validation set')

    pl.xlabel('Training sample size', fontsize=14)
    pl.ylabel(metric.__name__, fontsize=14)
    pl.tick_params(labelsize=14)
    pl.legend(fontsize=12)


def all_but(par, all_pars):
    """
    Given a str `par` and a list of str `all_pars`, returns a list of 
    all str in `all_pars` that are not `par`
    """
    return [p for p in all_pars if p != par]


def fix_other_best_par(par, best_par_loc):
    """
    Returns a bool Series that combines all columns in `best_par_loc` 
    boolean DataFrame, except `par`, with AND. 
    """
    return best_par_loc[all_but(par, best_par_loc.columns)].all(axis=1)


def weird_to_str(x):
    """
    If `x` is not float, int or str, convert to str.
    """
    if type(x) != float and type(x) != int and type(x) != str:
        return str(x)
    else:
        return x

    
def plot_scores_vs_par(slice_grid, par, score_name, logscale=True, random=False):
    """
    Given a DataFrame `slice_grid` with parameter `par` values and 
    train and test scores, plot the latter as a function of the former.
    """

    par_prefix     = 'param_'
    par_prefix_pos = len(par_prefix)
    
    x = slice_grid[par].apply(weird_to_str)
    
    #pl.plot(x, slice_grid['mean_train_score'], color='r', alpha=0.3)
    pl.errorbar(x, slice_grid['mean_train_score'], yerr=slice_grid['std_train_score'], 
                fmt='.', color='r', label='Training set')
    
    #pl.plot(x, slice_grid['mean_test_score'], color='b', alpha=0.3)
    pl.errorbar(x, slice_grid['mean_test_score'], yerr=slice_grid['std_test_score'], 
                fmt='.', color='b', label='Validation set')    
    
    if (logscale == True and x.dtype == np.float64) or logscale == 'force':
        pl.xscale('log')

    pl.legend(fontsize=12)
    pl.tick_params(labelsize=14)
    pl.ylabel(score_name, fontsize=14)
    pl.xlabel(par[par_prefix_pos:], fontsize=14)
    
    # Format tick labels in case of long strings:
    ## Get labels:
    pl.draw() 
    loc, labels = pl.xticks() # Aparentemente isso retorna marcações erradas. CORRIGIR!
    ## Find out wether it is a number (uses TeX) or not:
    if np.all([s.get_text().find('$') == -1 for s in labels]) == True:
        ## Crop strings:
        labels = [s.get_text()[:10] + '...' if len(s.get_text()) > 10 else s for s in labels]
    pl.gca().set_xticklabels(labels)
    
    

    
def plot_pars_scores(grid, logscale=True):
    """
    Given a GridSearchCV object `grid` already fitted to a training data,
    create one plot of training and validation accuracy vs. parameter 
    value for each parameter. All other parameters are held to their best 
    values. 
    
    If `logscale` is True (default), float parameters are plot in a log 
    scale.
    """
    par_prefix     = 'param_'
    par_prefix_pos = len(par_prefix)

    # Create dataframe from grid data:
    grid_runs = pd.DataFrame(grid.cv_results_)

    # Get columns of parameter values:
    param_columns = list(filter(lambda c: c[:par_prefix_pos] == par_prefix, grid_runs.columns))

    # Get best parameters:
    best_params = grid.best_params_

    # Build table for locating best parameters:
    best_par_loc = pd.DataFrame()
    for par in param_columns:
        best_par_loc[par] = grid_runs[par] == best_params[par[par_prefix_pos:]]

    # Get the name of the scoring:
    score_name = str(grid.get_params()['scoring'])
    if len(score_name) > 25:
        score_name = score_name[:25] + '...'
        
    # Numbers of plots (and if columns and rows):
    n_pars = len(param_columns)
    n_cols = 3
    n_rows = int((n_pars - 1)/ n_cols) + 1

    pl.figure(figsize=(5 * min(3, n_pars) , 5 * n_rows))
    for i, par in enumerate(param_columns):
        pl.subplot(n_rows, min(3, n_pars), i + 1)
        slice_grid = grid_runs.loc[fix_other_best_par(par, best_par_loc)]
        plot_scores_vs_par(slice_grid, par, score_name, logscale)

    pl.tight_layout()


#############################
### Saving models to disk ###
#############################

def end_slash(path):
    """
    Returns a standardized `path` with one slash in the end.
    """
    if path[-1] != '/':
        return path + '/'
    else:
        return path

    
def get_package_info(_package_name):
    """
    Returns a dict with the packages' name `_package_name`, its 
    version and its dependencies.
    """
    _package = pkg_resources.working_set.by_key[_package_name]

    version = _package.version
    
    dependencies = [str(r) for r in _package.requires()]

    return {'name': _package_name, 'version': version, 'dependencies': dependencies}


def save_package_versions(package_list, filename):
    """
    Given a list of python package names `package_list` and a JSON `filename`, 
    print the packages versions and dependencies to the file.
    """
    all_packages = []
    for package in package_list:
        all_packages.append(get_package_info(package))
    
    with open(filename, 'w') as f:
        json.dump(all_packages, f, indent=2)
    

def save_model(directory, tested_model, production_model, train_df, test_df, metric, train_score, test_score,  
               val_df=None, package_list=['scikit-learn', 'numpy', 'matplotlib', 'pandas', 'scipy', 'nltk']):
    """
    Save a machine-learning model to file, along with other information.
    
    
    Input
    -----
    
    directory : str
        The path to the directory where to save all the model's files and information.
    
    tested_model : python object 
        A model to be pickled, usually a sklearn Pipeline or analogous. This is the 
        model that was fit to the training data `train_df` and tested on the test 
        data `test_df`.
        
    production_model : python object
        A model to be pickled, usually a sklearn Pipeline or analogous. This is the 
        model that was fit to the concatenation of the training set `train_df` and 
        the test set `test_df` (and the validation set `val_df` if existing).
        
    train_df : Pandas DataFrame
        The data used to train the model `tested_model`, including both X and y 
        among its columns.
        
    test_df : Pandas DataFrame
        The data used to test the model `tested_model`, including both X and y
        among its columns.
        
    metric : str
        The name of the metric used to evaluate the model's performance.
    
    train_score : float
        The value of the metric obtained when applying it to the training set
        `train_df`.
        
    test_score : float
        The value of the metric obtained when applying it to the training set
        `test_df`.

    val_df : Pandas DataFrame (default None)
        Some models (such those for time evolving data) might require an extra validation
        set that is similar to the test set but cannot be extracted from the train set
        (because train set belongs to the past and validation and test sets belong to the 
        future). In these cases, this is the validation set, used to tune hypterparameters.

    package_list : list of str 
        A list of python packages for which to record the version and dependencies.
        It defaults to ['scikit-learn', 'numpy', 'matplotlib', 'pandas', 'scipy', 'nltk'].

    Outputs
    -------
    
    Files with hardcoded names, all written into `directory`.
    
    package_versions.json
        The versions of the packages installed when building the model, along with 
        their dependencies.
        
    tested_model_pars.txt
        The parameters of `tested_model` (the output of tested_model.get_params() 
        written out as a string).

    production_model_pars.txt
        The parameters of `production_model` (the output of production_model.get_params() 
        written out as a string).
        
    tested_model.joblib
        A pickled version of `tested_model`, written to a file using joblib.dump().
        It can be loaded using joblib.load().

    production_model.joblib
        A pickled version of `production_model`, written to a file using joblib.dump().
        It can be loaded using joblib.load().

    train_data.csv
        The `train_df` in CSV format.

    test_data.csv
        The `test_df` in CSV format.

    validation_data.csv
        The `val_df`  in CSV format, if provided.
           
    scores_and_info.json
        A JSON file containing the date when the model was saved, the metric used 
        to evaluate the model and the scores for the train and test sets.
    """
    
    # Create directory and format it:
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = end_slash(directory)
    
    # Pickle the model that was trained in the trained set and tested with the test set:
    joblib.dump(tested_model, directory + 'tested_model.joblib')

    # Pickle the model that was fit to the whole data:
    joblib.dump(production_model, directory + 'production_model.joblib')

    # Save train and test (and validation) data:
    train_df.to_csv(directory + 'train_data.csv', index=False)
    test_df.to_csv(directory + 'test_data.csv', index=False)
    if type(val_df) != type(None):
        val_df.to_csv(directory + 'validation_data.csv', index=False)
        
    # Save model parameters:
    with open(directory + 'tested_model_pars.txt', 'w') as f:
        f.write(str(tested_model.get_params()))
    with open(directory + 'production_model_pars.txt', 'w') as f:
        f.write(str(production_model.get_params()))
    
    # Save python package info:
    save_package_versions(package_list, directory + 'package_versions.json')
    
    # Save scores and other info:
    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    score_board = {'date': current_date, 'metric': metric, 'train_score': train_score, 'test_score': test_score}
    with open(directory + 'scores_and_info.json', 'w') as f:
        json.dump(score_board, f, indent=0)

