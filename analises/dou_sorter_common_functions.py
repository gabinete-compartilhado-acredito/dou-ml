#####################################################################
### Funções utilizadas no modelo de machine learning para ordenar ###
### matérias do Diário Oficial da União por relevância.           ###
#####################################################################

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

### Preprocessing       ###
### no fitting required ###


def tipo_edicao_Q(edicao):
    """
    Define se edição do artigo é ordinária ou extra.
    """
    return 'Extra' if len(edicao.split('-')) > 1 else 'Ordinária'


def get_article_type(string):
    """
    Parse article title (identifica) into a type (e.g. portaria, ato).
    """
    # List of specific replacements:
    speficic_replace = [('julgamento', 'julgamento '), ('adiamento', 'adiamento '), ('licitação', 'licitação '),
                        ('adjudicação', 'adjudicação '), ('homologação', 'homologação '), 
                        ('suspensão', 'suspensão '), ('prorrogação', 'prorrogação '), 
                        ('homologado', 'homologado '), ('retificação', 'retificação '), 
                        ('alteração', 'alteração '), ('revogação', 'revogação '), ('habilitação', 'habilitação '), 
                        ('pregão', ' pregão')]
    
    # If string is not str (e.g. NaN), return empty string:
    if type(string) != str:
        return ''
    
    # Process string:
    proc_string = string
    proc_string = proc_string.lower()
    # Remove excess of whitespace:
    proc_string = ' '.join(proc_string.strip().split())
    # Remove dates:
    proc_string = re.sub('de \d{1,2}[°º]? de [a-zç]{3,11} de \d{4}', '', proc_string)
    # Remove article numbering:
    proc_string = re.sub('n[°ºª] ?[0-9\.,/\-]*', '', proc_string)
    # Replace characters by space:
    proc_string = re.sub('/', ' ', proc_string)
    proc_string = re.sub('\|', ' ', proc_string)
    
    # Make specific replacements:
    for spec_rep in speficic_replace:
        proc_string = proc_string.replace(spec_rep[0], spec_rep[1])
    
    proc_string = ' '.join(proc_string.split()).strip()
    return proc_string


def count_levels(orgao, splitter):
    """
    Return the number of levels in the `orgao` (split by `splitter`).
    """
    return len(orgao.split(splitter))


def create_level_df(series, splitter='/', prefix='org'):
    """
    Split a Pandas `series` into columns of a dataframe using `splitter` 
    as the column separator. The lack of levels in a certain row 
    translates to empty columns on the right.
    """
    # Get number of levels:
    max_levels  = series.apply(lambda s: count_levels(s, splitter)).max()
    # Set names of columns:
    columns = [prefix + str(i) for i in range(1, max_levels + 1)]
    # Split Series into columns by splitter:
    splitted = pd.DataFrame(series.str.split(splitter).values.tolist(), columns=columns, index=series.index)
    
    return splitted


class PreprocessDOU(BaseEstimator, TransformerMixin):
    """
    Preprocess (no fitting required) a Pandas DataFrame containing information 
    about DOU publications. These are preprocessing specific to DOU, and 
    some combine or split columns. The steps are:

    -- Clean title to get only the kind of document;
    -- Find out from 'edicao' column if this is an ordinary or extra edition
       and set it as a new column;
    -- Set 'secao' as str;
    -- Fill missing values with `fillna`;
    -- Join columns containing texts (according to requested columns in 
       `colunas_relevantes`);
    -- Split `orgao` column into their hierarchy levels if 'orgaos' is 
       among `colunas_relevantes`.
    
    Input
    -----

    colunas_relevantes : list of str
        List of columns to keep in the output. These might be columns present in the 
        input DataFrame or new columns created by this transformer. To split column 
        'orgao' into columns 'org1', 'org2', ..., place 'orgaos' in `colunas_relevantes`.

    fillna : str
        A string used to identify that a value is missing.


    Output
    ------

    A Pandas DataFrame.
    """    

    def __init__(self, colunas_relevantes=None, fillna=None):
        self.colunas_relevantes = colunas_relevantes
        self.fillna = fillna
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        # Select all dataset:
        df = X.copy()            

        # PREPROCESS SOME COLUMNS:
        
        # Remove dates and numbers from title:
        if 'titulo' in self.colunas_relevantes:
            df['titulo'] = df.identifica.apply(get_article_type)
        
        # Is materia from ordinary or extra edition:
        if 'tipo_edicao' in self.colunas_relevantes:
            df['tipo_edicao'] = df.edicao.astype(str).apply(tipo_edicao_Q)
        
        # Transform seção to string:
        if 'secao' in self.colunas_relevantes:
            df['secao'] = df.secao.astype(str)
        
        # Fill missing values:
        if self.fillna != None:
            df = df.fillna(value=self.fillna)
        
        # Create a single text column with ementa and text:
        if 'ementa_text' in self.colunas_relevantes:
            df['ementa_text'] = df.ementa + ' ' + df.fulltext
        
        # Create a single text column with titulo, ementa and text:
        if 'tit_ementa_text' in self.colunas_relevantes:
            df['titulo'] = df.identifica.apply(get_article_type)
            df['tit_ementa_text'] = df.titulo + ' ' + df.ementa + ' ' + df.fulltext
        
        # Create a single text column with titulo, orgao, ementa and text:
        if 'tit_org_ementa_text' in self.colunas_relevantes:
            df['titulo'] = df.identifica.apply(get_article_type)
            orgao_text   = df.orgao.str.replace('/', ' ')
            df['tit_org_ementa_text'] = df.titulo + ' ' + orgao_text + ' ' + df.ementa + ' ' + df.fulltext

        # Create a single text column with orgao, ementa and text:
        if 'org_ementa_text' in self.colunas_relevantes:
            orgao_text   = df.orgao.str.replace('/', ' ')
            df['org_ementa_text'] = orgao_text + ' ' + df.ementa + ' ' + df.fulltext
            
        # Split orgaos by their level:
        if 'orgaos' in self.colunas_relevantes:
            # Split orgaos by '/':
            orgaos = create_level_df(df.orgao)
            # Transform 'orgaos' column into level columns:
            self.colunas_relevantes = list(filter(lambda s: s != 'orgaos', self.colunas_relevantes))
            self.colunas_relevantes = self.colunas_relevantes + list(orgaos.columns)
            # Fill missing values in orgaos: 
            if self.fillna != None:
                orgaos = orgaos.fillna(value=self.fillna)
            df = df.join(orgaos)
            assert len(df.dropna(how='any')) == len(X), 'Number of rows is not preserved'
        
        # Only output selected columns (default: passthrough):
        df = df[self.colunas_relevantes]

        return df


def remove_punctuation(text, keep_cash=True):
    """
    Remove punctuation from text.
    
    Input
    -----
    
    text : str
        Text to remove punctuation from.
        
    keep_cash : bool (default True)
        Whether to remove the dollar sign '$' or not.
    """
    # Define which punctuation to remove:
    punctuation = '!"#%&\'()*+,-.:;<=>?@[\\]^_`{|}~'
    if keep_cash == False:
        punctuation = punctuation + '$'
    
    return text.translate(str.maketrans('/', ' ', punctuation))


def remove_stopwords(text, stopwords):
    """
    Remove list of words (str) in `stopwords` from string `text`. 
    """
    word_list = text.split()
    word_list = [word for word in word_list if not word in set(stopwords)]
    return ' '.join(word_list)


def stem_words(text, stemmer):
    """
    Given a `stemmer`, use it to stem `text`.  
    """
    #stemmer   = nltk.stem.RSLPStemmer()   # Português
    #stemmer   = nltk.stem.PorterStemmer() # Inglês
    word_list = text.split()
    word_list = [stemmer.stem(word) for word in word_list]
    return ' '.join(word_list)


def remove_accents(string, i=0):
    """
    Input: string
    
    Returns the same string, but without all portuguese-valid accents.
    """
    accent_list = [('Ç','C'),('Ã','A'),('Á','A'),('À','A'),('Â','A'),('É','E'),('Ê','E'),('Í','I'),('Õ','O'),
                   ('Ó','O'),('Ô','O'),('Ú','U'),('Ü','U'),('ç','c'),('ã','a'),('á','a'),('à','a'),('â','a'),
                   ('é','e'),('ê','e'),('í','i'),('õ','o'),('ó','o'),('ô','o'),('ú','u'),('ü','u')]
    if i >= len(accent_list):
        return string
    else:
        string = string.replace(*accent_list[i])
        return remove_accents(string, i + 1)

    
def keep_only_letters(text, keep_cash=True):
    """
    Remove from string `text` all characters that are not letters (letters include those 
    with portuguese accents). If `keep_cash` is true, do not remove the dollar sign 
    '$'.
    """
    if keep_cash == True:
        extra_chars = '$'
    else:
        extra_chars = ''
        
    only_letters = re.sub('[^a-z A-ZÁÂÃÀÉÊÍÔÓÕÚÜÇáâãàéêíóôõúüç' + extra_chars + ']', '', text)
    only_letters = ' '.join(only_letters.split())
    return only_letters


def num_to_scale(x, b=2):
    """
    Given a float `x`, returns its log on base `b`.
    """
    return int(np.round(np.log(np.clip(x, a_min=1, a_max=None)) / np.log(b)))


def number_scale_token(matchobj):
    """
    Given a regex match object that is supposed to match a number in 
    Brazilian format (e.g. 10.643.543,05), replace it by a string token 
    whose length is proportional to its scale (i.e. log).
    """
    full_match   = matchobj.group(0)
    target_match = matchobj.group(1)
    
    target_float = float(target_match.replace('.', '').replace(',', '.'))
    target_new   = ' xx' + 'v' * num_to_scale(target_float) + 'xx '    
    full_new     = full_match.replace(target_match, target_new)
    return full_new


def values_to_token(text, regex_list):
    """
    Given a string `text` and a list of regex patterns for values 
    in Brazilian format (e.g. 24.532,78), replace them by a string 
    token whose length is prop. to the log of the value.
    """
    tokenized = text
    for regex in regex_list:
        tokenized = re.sub(regex, number_scale_token, tokenized)
        
    return tokenized


def tuple_regex_sub(pattern_token, text):
    """
    Replace a regex with a token in text.
    
    Input
    -----
    
    pattern_token : tuple of str
        The tuple is (regex, token), where `regex` is the regular expression
        to be replaced for `token`.
        
    text : str
        String where the replacement occurs.
    """
    return re.sub(pattern_token[0], pattern_token[1], text, flags=re.I)


def tokenize_cargos(regex_token, text, verbose=False):    
    """
    Replace references to posts (cargos) in `text` with standardized tokens.
    The cargos are hard-coded as regex patterns.
    """
    
    result = text
    for rt in regex_token: 
        result = tuple_regex_sub(rt, result)
    
    return result


class PreProcessText(BaseEstimator, TransformerMixin):
    """
    Preprocess text (no fitting required) stored in a DataFrame columns.
    
    Given a list of columns `text_cols`, apply a series of transformations 
    to all of them, as requested by the instance parameters:
    
    -- Set all to lowercase;
    -- Transform numbers representing R$ amounts into tokens 
       according to their scale;
    -- Remove punctuation from text;
    -- Do not remove the dollar sign '$';
    -- Remove stopwords (list of str);
    -- Use `stemmer` to stem the words;
    -- Remove accents; 
    -- Keep only letters (and the dollar sign, if requested).
    
    Return
    ------
    
    Pandas DataFrame.
    """
    def __init__(self, text_cols=None, lowercase=True, value_tokens=True, remove_punctuation=True, 
                 keep_cash=True, stopwords=None, stemmer=None, strip_accents=True, only_letters=True,
                 cargo_tokens=True):
        self.text_cols          = text_cols
        self.lowercase          = lowercase
        self.value_tokens       = value_tokens
        self.remove_punctuation = remove_punctuation
        self.keep_cash          = keep_cash
        self.stopwords          = stopwords
        self.stemmer            = stemmer
        self.strip_accents      = strip_accents
        self.only_letters       = only_letters
        self.cargo_tokens       = cargo_tokens
        
        # Value to token regex:
        regex1_str = r'[rR]\$ ?(\d{1,3}(?:\.\d{3}){0,4}\,?\d{0,2})'
        regex2_str = r'(\d{1,3}(?:\.\d{3}){0,4}\,?\d{0,2}) (?:reais|REAIS|Reais)'
        regex3_str = r'(\d{1,3}(?:\.\d{3}){0,4},\d{2})(?:[^%]|$)'
        regex1 = re.compile(regex1_str)
        regex2 = re.compile(regex2_str)
        regex3 = re.compile(regex3_str)
        self.regex_list = [regex1, regex2, regex3]
        
        # Compile post (cargos) regexes:
        self.cargo_regex = [(r'das[ -]*?[0123]{3}\.6', ' xxdasseisxx '),
                       (r'das[ -]*?[0123]{3}\.5', ' xxdascincoxx '),
                       (r'das[ -]*?[0123]{3}\.4', ' xxdasquatroxx '),
                       (r'das[ -]*?[0123]{3}\.3', ' xxdastresxx '),
                       (r'das[ -]*?[0123]{3}\.2', ' xxdasdoisxx '),
                       (r'das[ -]*?[0123]{3}\.1', ' xxdasumxx '),
                       (r'\Wca[ -]+?i(?:\W|$)', r' xxcaixx '),
                       (r'\Wca[ -]+?ii(?:\W|$)', r' xxcaiixx '),
                       (r'ca-apo[ -]*?1', ' xxcaapoumxx '),
                       (r'ca-apo[ -]*?2', ' xxcaapodoisxx '),
                       (r'\Wcdt\W', ' xxcdtxx '),
                       (r'ccd[ -]+?i(?:\W|$)', ' xxccdixx '),
                       (r'ccd[ -]+?ii(?:\W|$)', ' xxccdiixx '),
                       (r'cge[ -]+?i(?:\W|$)', ' xxcgeixx '),
                       (r'cge[ -]+?ii(?:\W|$)', ' xxcgeiixx '),
                       (r'cge[ -]+?iii(?:\W|$)', ' xxcgeiiixx '),
                       (r'cge[ -]+?iv(?:\W|$)', ' xxcgeivxx '),
                       (r'cge[ -]+?v(?:\W|$)', ' xxcgevxx '),
                       (r'cpaglo', ' xxcpagloxx '),
                       (r'\W(csp)(?:\W|$)', ' xxcspxx '),
                       (r'\W(csu)(?:\W|$)', ' xxcsuxx '),
                       (r'\Wcd(?:[ -]*?|\.)1(?:\W|$)', ' xxcdumxx '), 
                       (r'\Wcd(?:[ -]*?|\.)2(?:\W|$)', ' xxcddoisxx '),
                       (r'\Wne(?:\W|$)', ' xxnexx '),
                       (r'cetg[ -]*?iv(?:\W|$)', ' xxcetgivxx '),
                       (r'cetg[ -]*?v(?:\W|$)', ' xxcetgvxx '),
                       (r'cetg[ -]*?vi(?:\W|$)', ' xxcetgvixx '),
                       (r'cetg[ -]*?vii(?:\W|$)', ' xxcetgviixx '),
                       (r'\Wfds[ -]*?1(?:\W|$)', ' xxfdsumxx '), 
                       (r'fc?pe[ -]*?[0-9]{3}\.5', ' xxfcpecincoxx '),
                       (r'fc?pe[ -]*?[0-9]{3}\.4', ' xxfcpequatroxx '),
                       (r'fc?pe[ -]*?[0-9]{3}\.3', ' xxfcpetresxx '),
                       (r'fc?pe[ -]*?[0-9]{3}\.2', ' xxfcpedoisxx '),
                       (r'fc?pe[ -]*?[0-9]{3}\.1', ' xxfcpeumxx '),
                       ('natureza especial', ' xxnaturezaespecialxx '),
                       (r'cne[ -]*?([0-9]{2})', r' xxcne\1xx ')]
        #self.cargo_regex = [(re.compile(r), t) for r,t in cargo_regex]
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Apply transforms to all X columns that are text:
        if self.text_cols != None:
            for col in self.text_cols:
                # Tokenize cargos:
                if self.cargo_tokens:
                    df[col] = df[col].apply(lambda s: tokenize_cargos(self.cargo_regex, s))
                # Lowercase:
                if self.lowercase:
                    df[col] = df[col].str.lower()
                # Transform R$ values to tokens:
                if self.value_tokens:
                    df[col] = df[col].apply(lambda s: values_to_token(s, self.regex_list))
                # Remove punctuation:
                if self.remove_punctuation:
                    df[col] = df[col].apply(lambda s: remove_punctuation(s, self.keep_cash))
                # Remove stopwords:
                if self.stopwords != None:
                    df[col] = df[col].apply(lambda s: remove_stopwords(s, self.stopwords))
                # Stem words:
                if self.stemmer != None:
                    df[col] = df[col].apply(lambda s: stem_words(s, self.stemmer))
                # Remove accents:
                if self.strip_accents:
                    df[col] = df[col].apply(remove_accents)
                # Keep only leters:
                if self.only_letters:
                    df[col] = df[col].apply(lambda s: keep_only_letters(s, self.keep_cash))
        return df


#############################################
### Custom Transformers (fitting required ###
#############################################


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
        super().__init__(input, encoding, decode_error, strip_accents, lowercase, preprocessor, tokenizer, 
                         stop_words, token_pattern, ngram_range, analyzer, max_df, min_df, max_features, 
                         vocabulary, binary, dtype)
        
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



######################    
### Model builders ###
######################


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

def create_model_sec1():
    
    # Pré-processamento (s/ fit):
    text_col = 'tit_org_ementa_text'
    colunas_relevantes = ['tipo_edicao'] + [text_col]

    stopwords = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
                 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também',
                 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse',
                 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles',
                 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus',
                 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles',
                 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava',
                 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos',
                 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera',
                 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei',
                 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram',
                 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos',
                 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos',
                 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos',
                 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam']

    dou_extractor = PreprocessDOU(colunas_relevantes, ' xxnuloxx ')

    proc_text = PreProcessText(lowercase=False, remove_punctuation=True, keep_cash=True, stopwords=stopwords, 
                              stemmer=None, strip_accents=False, only_letters=False, cargo_tokens=True,
                               text_cols=[text_col])

    # Fit processing and model:
    #vectorizer    = CountVectorizer(lowercase=False, binary=True, ngram_range=(1,2), max_df=1.0, min_df=1)
    vectorizer    = TfidfVectorizer(lowercase=False, binary=True, ngram_range=(1,2), max_df=1.0, min_df=1, norm=None, use_idf=True)
    
    encoder_extra = OneHotEncoder(drop='first')
    processor     = ColumnTransformer([('vec',   vectorizer,    text_col),
                                       ('extra', encoder_extra, ['tipo_edicao'])
    ])

    classifier  = Ridge(10000)

    pipeline = Pipeline([('dou', dou_extractor), ('pretext', proc_text), ('proc', processor), ('fit', classifier)])

    return pipeline



def create_model_sec2():
    
    # Pré-processamento (s/ fit):
    text_col = 'tit_org_ementa_text'
    colunas_relevantes = ['tipo_edicao'] + [text_col]

    stopwords = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
                 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também',
                 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse',
                 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles',
                 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus',
                 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles',
                 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava',
                 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos',
                 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera',
                 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei',
                 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram',
                 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos',
                 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos',
                 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos',
                 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam']

    dou_extractor = PreprocessDOU(colunas_relevantes, ' xxnuloxx ')
    

    proc_text = PreProcessText(cargo_tokens=True, lowercase=True, remove_punctuation=True, keep_cash=True, 
                              stopwords=stopwords, stemmer=None, strip_accents=False, only_letters=False,
                              text_cols=[text_col])
    

    # Fit processing and model:
    vectorizer    = CountVectorizer(lowercase=False, binary=True, ngram_range=(1,1), max_df=0.7, min_df=2)
    encoder_extra = OneHotEncoder(drop='first')
    processor     = ColumnTransformer([('vec',   vectorizer,    text_col),
                                       ('extra', encoder_extra, ['tipo_edicao'])])

    #classifier  = Ridge(20)
    classifier = VotingRegressor([('ridge', Ridge(30)), 
                                  ('forest', RandomForestRegressor(max_depth=25, min_samples_split=2, n_estimators=20, min_samples_leaf=1, max_samples=0.7, n_jobs=7))])

    pipeline = Pipeline([('dou', dou_extractor), ('pretext', proc_text), ('proc', processor), ('fit', classifier)])

    return pipeline


