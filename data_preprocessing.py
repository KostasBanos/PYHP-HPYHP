import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from collections import Counter
from collections import OrderedDict
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
import heapq
import collections
import datetime
import time
from subprocess import check_output
import os

print("Available data files:")
print(os.listdir("./data"))


# =============================================================================
# UTILITY FUNCTIONS FOR DATA PROCESSING
# =============================================================================

# Set vocabulary size limit for efficient processing
dictionarySize = 3000

def downloadNews():
    """
    Load and combine news articles from multiple CSV files.
    
    This function reads three separate CSV files containing news articles
    and combines them into a single DataFrame with continuous indexing.
    The articles come from different sources and time periods.
    
    Returns:
        pd.DataFrame: Combined news articles with columns: id, title, publication, 
                     author, date, year, month, url, content
    """
    # Load individual CSV files
    news1 = pd.read_csv('./data/articles1.csv')
    news2 = pd.read_csv('./data/articles2.csv')
    news3 = pd.read_csv('./data/articles3.csv')
    
    # Create continuous indexing across all datasets
    news1.index = range(0, news1.shape[0])
    news2.index = range(news1.shape[0], news1.shape[0] + news2.shape[0])
    news3.index = range(news1.shape[0] + news2.shape[0], 
                       news1.shape[0] + news2.shape[0] + news3.shape[0])
    
    # Combine all datasets into single DataFrame
    news = pd.concat([news1, news2, news3])
    return news

def getWordList(someContent, tupleLength):
    """
    Generate n-grams (word tuples) from text content.
    
    This function creates sliding window n-grams from the input text,
    which can be used for language modeling or text analysis.
    
    Args:
        someContent (list): List of words/tokens
        tupleLength (int): Length of n-grams to generate
        
    Returns:
        list: List of n-gram strings
    """
    t = tupleLength
    aList = []
    
    # Create sliding windows of specified length
    for i in range(0, tupleLength):
        aList.append(someContent[i:len(someContent) - (t - i)])
    
    # Transpose to get n-grams
    trans = list(map(list, zip(*(aList))))
    return list(map(''.join, trans))

def getAgenciesContent():
    """
    Organize news articles by publication agency.
    
    This function groups articles by their publication source (e.g., 
    New York Times, Washington Post) for agency-specific analysis.
    
    Returns:
        dict: Dictionary mapping agency names to their article content lists
    """
    news = downloadNews()
    Agencies = news.publication.unique()  # Get unique publication names
    AgencyContent = {}
    
    # Group articles by publication
    for aAgency in Agencies:
        AgenRows = news.loc[news['publication'] == aAgency]
        AgencyContent[aAgency] = AgenRows['content']
    
    return AgencyContent

def getPredictionTuples(someContent, tupleLength):
    """
    Generate prediction tuples for language modeling.
    
    Creates input-output pairs where the input is an n-gram and the output
    is the next word, useful for training language models.
    
    Args:
        someContent (list): List of words/tokens
        tupleLength (int): Length of input n-grams
        
    Returns:
        list: List of (input_ngram, next_word) tuples
    """
    prediction = someContent[tupleLength:len(someContent) - 1]
    wordList = getWordList(someContent, tupleLength)
    return tuple(map(tuple, zip(*[wordList, prediction])))

def getContentDictionary(someContent, tupleLength):
    """
    Build word frequency dictionary from content.
    
    Processes multiple articles to create a comprehensive word frequency
    dictionary, useful for vocabulary analysis and text preprocessing.
    
    Args:
        someContent (list): List of article contents
        tupleLength (int): Length of n-grams to extract
        
    Returns:
        tuple: (Counter object with word frequencies, list of all words)
    """
    otherChars = string.punctuation + ' ' + ''
    totalCount = Counter()
    bigWordList = []
    
    # Process each article
    for aArticle in someContent:
        # Split text into tokens using regex
        aSplitList = list(re.split('(\\W)', aArticle.lower()))
        # Remove punctuation and whitespace
        aCleanSplitList = [chunk for chunk in aSplitList if chunk not in otherChars]
        # Extract n-grams
        wordList = getWordList(aCleanSplitList, tupleLength)
        
        # Accumulate words and counts
        bigWordList.extend(wordList)
        aCount = Counter(wordList)
        totalCount = aCount + totalCount
    
    return totalCount, bigWordList

def getSorted(aList):
    """
    Sort list by count in descending order.
    
    Args:
        aList (list): List of (item, count) tuples
        
    Returns:
        list: Sorted list by count (highest first)
    """
    def getCount(item):
        return -item[1]  # Negative for descending order
    return sorted(aList, key=getCount)

def plotWordDict(tempDict):
    """
    Create bar plot of word frequencies.
    
    Args:
        tempDict (dict): Dictionary of word frequencies
    """
    tempDict = dict(tempDict)
    objects = tempDict.keys()
    y_pos = np.arange(len(objects))
    performance = tempDict.values()

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Frequency')
    plt.title('Word Frequency Distribution')
    plt.show()

def getEmbedding(wordCol, aWord):
    """
    Create one-hot encoding for a word.
    
    Args:
        wordCol (list): Vocabulary list
        aWord (str): Word to encode
        
    Returns:
        np.array: One-hot encoded vector
    """
    hotEncod = np.full((1, len(wordCol)), 0)
    if aWord in wordCol:
        index = wordCol.index(aWord)
        hotEncod[0][index] = 1
    else:
        hotEncod[0][len(wordCol) - 1] = 1  # Unknown word position
    return hotEncod

def getEmbeddingIndex(wordCol, aWord):
    """
    Get index of word in vocabulary.
    
    Args:
        wordCol (list): Vocabulary list
        aWord (str): Word to find
        
    Returns:
        int: Index of word, or last index for unknown words
    """
    if aWord in wordCol:
        return wordCol.index(aWord)
    else:
        return len(wordCol) - 1  # Unknown word index

def getWordFromEmbeding(wordCol, aWordVector):
    """
    Convert one-hot vector back to word.
    
    Args:
        wordCol (list): Vocabulary list
        aWordVector (np.array): One-hot encoded vector
        
    Returns:
        str: Corresponding word
    """
    return getWordFromIndex(wordCol, np.nonzero(aWordVector == 1)[1][0])

def getWordFromIndex(wordCol, wordIndex):
    """
    Get word from vocabulary index.
    
    Args:
        wordCol (list): Vocabulary list
        wordIndex (int): Index in vocabulary
        
    Returns:
        str: Word at given index, or special token for unknown
    """
    if wordIndex < len(wordCol) - 1:
        outWord = wordCol[wordIndex]
    else:
        outWord = './RareWord'  # Special token for unknown words
    return outWord

print('Utility functions loaded successfully')


# =============================================================================
# LOAD AND EXAMINE NEWS DATA
# =============================================================================

# Load the combined news dataset
news = downloadNews()
print(f"Dataset shape: {news.shape}")
print(f"Total articles: {news.shape[0]}")
print(f"Columns: {list(news.columns)}")

# Display first few rows to understand data structure
print("\nFirst 5 articles:")
news.head()

# =============================================================================
# TEXT PREPROCESSING SETUP
# =============================================================================

# Load English stopwords for text filtering
with open('data/english.stop.txt', 'r') as f:
    english_stop = f.read().splitlines()

# Convert to set for efficient lookup
english_stop = set(english_stop)

# Manually add some specific stopwords that might appear in news articles
english_stop.add('ms')  # Miss
english_stop.add('mr')   # Mister

def hasNumbers(inputString):
    """
    Check if a string contains any digits.
    
    Args:
        inputString (str): String to check
        
    Returns:
        bool: True if string contains digits, False otherwise
    """
    return any(char.isdigit() for char in inputString)

def text_preprocess(text):
    """
    Comprehensive text preprocessing pipeline.
    
    This function performs the following preprocessing steps:
    1. Remove non-printable characters
    2. Convert to lowercase
    3. Tokenize using regex
    4. Remove punctuation and stopwords
    5. Remove words containing numbers
    6. Apply Porter stemming
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        list: List of preprocessed tokens
    """
    # Define characters to remove
    otherChars = string.punctuation + ' ' + ''
    printable = set(string.printable)
    
    # Remove non-printable characters
    text = ''.join(filter(lambda x: x in printable, text))
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using regex (split on non-word characters)
    token_list = list(re.split('(\\W)', text.lower()))
    
    # Initialize Porter stemmer for word normalization
    ps = PorterStemmer()
    # Note: WordNetLemmatizer is available but not used in this implementation
    
    # Apply filtering and stemming
    return [ps.stem(token) for token in token_list 
            if (token not in otherChars) and 
               (token not in english_stop) and 
               (not hasNumbers(token))]


# =============================================================================
# TEXT PROCESSING AND VOCABULARY BUILDING
# =============================================================================

# Initialize data structures for processed content
vocabulary = set()  # Set to store unique words across all documents
dates = []          # List to store publication dates
texts = []          # List to store preprocessed text for each document

# Process each news article
for i in range(news.shape[0]):
    # Extract and preprocess the article content
    text = text_preprocess(news.loc[i, :]['content'])
    date = news.loc[i, :]['date']
    
    # Update vocabulary with new words from this document
    vocabulary = vocabulary | set(text)
    
    # Store processed data
    texts.append(text)
    dates.append(date)
    
    # Progress indicator for large datasets
    if (i + 1) % 1000 == 0:
        print(f'Finished processing document {i}')
    
    # Limit processing for demonstration (remove this to process all data)
    # if i >= 1000:  
        # break
       

# =============================================================================
# VOCABULARY SIZE ANALYSIS
# =============================================================================

# Display the total vocabulary size after processing
print(f"Total vocabulary size: {len(vocabulary)}")
print("This represents the number of unique words found across all processed documents.")


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

# Clear the large news DataFrame from memory to free up resources
# This is important for memory management when processing large datasets
import gc
news = None
gc.collect()

print("Memory cleanup completed. News DataFrame cleared from memory.")


# =============================================================================
# VOCABULARY FILTERING AND MAPPING CREATION
# =============================================================================

# Count word frequencies across all documents
cnt = Counter(word for text in texts for word in set(text))

# Filter vocabulary based on frequency thresholds
# This removes very rare words (frequency <= 5) and very common words (frequency >= 35256)
# This helps reduce noise and focus on meaningful vocabulary
cnt = Counter({k: v for k, v in cnt.items() if v > 5 and v < 35256})

# Create bidirectional word-to-ID mappings
word2id = {}  # Maps words to their integer IDs
id2word = {}  # Maps integer IDs back to words

# Build the mappings
for idx, (word, count) in enumerate(cnt.items()):
    word2id[word] = idx
    id2word[idx] = word

print(f"Filtered vocabulary size: {len(word2id)}")
print("Created word2id and id2word mappings for efficient text processing.")


# =============================================================================
# SAVE VOCABULARY MAPPINGS
# =============================================================================

# Save the word-to-ID mappings to JSON files for later use
# This allows the mappings to be reused in other parts of the pipeline
import json

# Save word2id mapping
with open('./data/word2id.json', 'w') as f:
    json.dump(word2id, f)

# Save id2word mapping    
with open('./data/id2word.json', 'w') as f:
    json.dump(id2word, f)

print("Vocabulary mappings saved to:")
print("- ./data/word2id.json")
print("- ./data/id2word.json")


# =============================================================================
# CONVERT TEXTS TO BAG-OF-WORDS REPRESENTATION
# =============================================================================

# Convert preprocessed texts into bag-of-words format suitable for
# the Sequential Monte Carlo Hawkes Process algorithm
news_items = []

for i, text in enumerate(texts):
    # Count word frequencies in this document
    text_cnt = Counter(text)
    words = text_cnt.keys()
    
    # Create word ID to count mapping for this document
    id2count_per_doc = {}
    for word in words:
        try:
            # Get word ID from vocabulary
            word_id = word2id[word]
            id2count_per_doc[word_id] = text_cnt[word]
        except KeyError:
            # Skip words not in filtered vocabulary
            continue
    
    # Convert to tuple format
    id2count_per_doc = id2count_per_doc.items()
    
    try:
        # Extract word IDs and counts
        word_ids, counts = zip(*id2count_per_doc)
        word_distribution = (word_ids, counts)
        word_count = sum(counts)
        date = dates[i]
        
        # Only include documents with sufficient word count (> 50 words)
        # This filters out very short articles that might not be meaningful
        if word_count > 50:
            news_items.append([date, word_distribution, word_count])
    except ValueError:
        # Skip documents with no valid words
        pass

print(f"Created {len(news_items)} news items with bag-of-words representation")
print("Each item contains: [date, (word_ids, counts), total_word_count]")


# =============================================================================
# VERIFY BAG-OF-WORDS CONVERSION
# =============================================================================

# Check the number of news items created
print(f"Total news items after bag-of-words conversion: {len(news_items)}")
print("This count represents articles that:")
print("- Have more than 50 words")
print("- Contain words from the filtered vocabulary")
print("- Have valid date information")


# =============================================================================
# DATE PARSING UTILITY
# =============================================================================

def try_parsing_date(text):
    """
    Parse date string in various formats.
    
    This function attempts to parse date strings in common formats
    used in news datasets. It handles both YYYY-MM-DD and YYYY/MM/DD formats.
    
    Args:
        text (str): Date string to parse
        
    Returns:
        datetime.date: Parsed date object
        
    Raises:
        ValueError: If no valid date format is found
    """
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError('no valid date format found')



# =============================================================================
# DATE VALIDATION AND FILTERING
# =============================================================================

# Parse dates and filter out invalid entries
dated_news_item = []

for i, news_item in enumerate(news_items):
    try:
        # Extract components from news item
        datestr = news_item[0]
        word_distribution = news_item[1]
        word_count = news_item[2]
        
        # Parse the date string
        date = try_parsing_date(datestr)
        
        # Create new item with parsed date
        dated_news_item.append([date, word_distribution, word_count])
        
    except TypeError:
        # Handle cases where date is not a string
        print(f"Invalid date format at index {i}: {datestr}")



# =============================================================================
# TEMPORAL SORTING
# =============================================================================

# Sort news items chronologically by date
# This is crucial for the Sequential Monte Carlo algorithm which processes
# events in temporal order to model temporal dependencies
sorted_news_items = sorted(dated_news_item, key=lambda x: x[0])

print(f"Sorted {len(sorted_news_items)} news items chronologically")
print("Items are now ordered by publication date for temporal analysis")


# =============================================================================
# VERIFY TEMPORAL SORTING
# =============================================================================

# Confirm the sorting was successful
print(f"Total sorted news items: {len(sorted_news_items)}")
print("All items are now ordered chronologically for temporal analysis")


# =============================================================================
# FILTER FOR 2017 DATA
# =============================================================================

# Focus on 2017 data for the analysis
# This creates a more focused dataset for the Sequential Monte Carlo algorithm
begin_date = datetime.date(2017, 1, 1)
news_items_2017 = [news_item for news_item in sorted_news_items 
                   if news_item[0] >= begin_date]

print(f"Filtered to {len(news_items_2017)} news items from 2017")
print("This focused dataset will be used for the Hawkes Process analysis")


# =============================================================================
# TEMPORAL TIMESTAMP GENERATION
# =============================================================================

def date2unixtime(date):
    """
    Convert date object to Unix timestamp.
    
    Args:
        date (datetime.date): Date object to convert
        
    Returns:
        float: Unix timestamp (seconds since epoch)
    """
    unixtime = time.mktime(date.timetuple())
    return unixtime

def date_conversion_distinctive_timestamp(date_seq):
    """
    Generate distinctive timestamps for documents published on the same date.
    
    This function is crucial for the Hawkes Process as it ensures each document
    has a unique timestamp, even when multiple articles are published on the same day.
    It distributes timestamps evenly throughout each day.
    
    Args:
        date_seq (list): List of date objects
        
    Returns:
        list: List of unique Unix timestamps
    """
    # Count documents per date
    date_cnt = Counter(date_seq)
    ordered_dates = sorted(date_cnt.keys())
    timestamps = []
    
    for date in ordered_dates:
        # Convert date to Unix timestamp
        date_in_unixtime = date2unixtime(date)
        
        # Calculate next day timestamp
        next_date = date + datetime.timedelta(days=1)
        doc_num_in_day = date_cnt[date]
        next_day_in_unixtime = date2unixtime(next_date)
        
        # Distribute timestamps evenly throughout the day
        # This ensures each document has a unique timestamp
        daily_timestamps = np.linspace(date_in_unixtime, next_day_in_unixtime, 
                                      num=doc_num_in_day, endpoint=False)
        daily_timestamps = daily_timestamps.tolist()
        
        # Verify timestamps are within the day
        assert daily_timestamps[-1] < next_day_in_unixtime
        
        timestamps.extend(daily_timestamps)
    
    return timestamps

# Generate distinctive timestamps for 2017 data
date_seq = [news_item[0] for news_item in news_items_2017]
timestamps = date_conversion_distinctive_timestamp(date_seq)

print(f"Generated {len(timestamps)} distinctive timestamps")
print("Each document now has a unique temporal identifier")


# =============================================================================
# VERIFY TEMPORAL ORDERING
# =============================================================================

# Ensure timestamps are in strictly increasing order
# This is essential for the Hawkes Process which relies on temporal ordering
for i, _ in enumerate(timestamps):
    if i == 0:
        continue
    assert timestamps[i] > timestamps[i-1], (i, timestamps[i], timestamps[i-1])

print("✓ Verified: All timestamps are in strictly increasing order")
print("This ensures proper temporal modeling for the Hawkes Process")


# =============================================================================
# CREATE FINAL DATA STRUCTURE
# =============================================================================

# Combine all components into the final format required by the SMC algorithm
# Format: (index, timestamp, word_distribution, word_count)
date, word_distribution, word_count = zip(*news_items_2017)
idx = range(len(news_items_2017))
news_items_2017 = zip(idx, timestamps, word_distribution, word_count)

print("Created final data structure with format:")
print("(document_index, unix_timestamp, (word_ids, counts), total_word_count)")
print(f"Total items: {len(list(news_items_2017))}")


# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

# Save the final processed dataset to JSON format
# This file will be used by the Sequential Monte Carlo Hawkes Process algorithm
import json

# Convert to list for JSON serialization (Python 3.7+ compatibility)
news_items_2017 = list(zip(idx, timestamps, word_distribution, word_count))

# Save to JSON file
with open('./data/all_the_news_2017.json', 'w') as f:
    json.dump(news_items_2017, f)

print("✓ Saved processed data to: ./data/all_the_news_2017.json")
print("This file contains the final dataset ready for SMC analysis")
print(f"Dataset includes {len(news_items_2017)} news articles from 2017")



# =============================================================================
# FINAL SUMMARY
# =============================================================================

# Display final vocabulary size and processing summary
print("=" * 60)
print("DATA PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Final vocabulary size: {len(id2word)}")
print(f"Total processed articles: {len(news_items_2017)}")
print(f"Time period: 2017")
print(f"Output files created:")
print(f"  - ./data/word2id.json (word-to-ID mapping)")
print(f"  - ./data/id2word.json (ID-to-word mapping)")
print(f"  - ./data/all_the_news_2017.json (processed dataset)")
print("=" * 25)
print("The dataset is now ready!")
print("=" * 25)