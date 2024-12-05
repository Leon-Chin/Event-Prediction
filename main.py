from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling1D, Conv1D, Flatten, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gensim.downloader as api
from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors
import nltk
import emoji


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# cache
# 创建 cache 文件夹
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

VECTOR_SIZE = 200

GLOVE_200_CACHE_PATH = os.path.join(
    CACHE_DIR,  "glove_twitter_200_cache.kv")

PREPROCESSED_TWEETS_CACHE_PATH = os.path.join(
    CACHE_DIR, "preprocessed_train_tweets.csv")

GLOVE_TWEET_VECTORS_CACHE_PATH = os.path.join(
    CACHE_DIR, "glove_tweet_vectors.npy")

PERIOD_FEATURES_CACHE_PATH = os.path.join(CACHE_DIR, "period_features.parquet")

# Load GloVe embeddings
print("start")


class Preprocessor:
    def __init__(self, glove_path=GLOVE_200_CACHE_PATH):
        self.glove_path = glove_path
        self.model = None
        self.original_data = None

    def process(self):
        self.load_glove_model()
        if self.original_data is None:
            self.load_orginal_data_and_preprocess_tweets()
        self.process_tweet()
        return self.original_data

    def load_orginal_data_and_preprocess_tweets(self, cache_path=PREPROCESSED_TWEETS_CACHE_PATH):
        if os.path.exists(cache_path):
            print("Loading preprocessed data from cache...")
            df = pd.read_csv(cache_path)
            self.original_data = df
        else:
            li = []
            print("reading file")
            for filename in os.listdir("train_tweets"):
                df = pd.read_csv("train_tweets/" + filename)
                li.append(df)

            print("finished read file")
            print("--------")
            df = pd.concat(li, ignore_index=True)
            print(df.head())
            print("Preprocessing the tweet text~")
            for i, tweet in enumerate(df['Tweet']):
                df.at[i, 'Tweet'] = self.preprocess_text(tweet)
                if i % 10000 == 0:  # 每处理 10000 条打印进度
                    print(f"Processed {i}/{len(df)} tweets...")
            print("Preprocessing complete!")

            # save it to cache file
            df = df.drop_duplicates(
                subset=['MatchID', 'PeriodID', 'Tweet'], keep='first')
            df.to_csv(cache_path, index=False)
            self.original_data = df
            print(f"Preprocessed data saved to {cache_path}")

        print("Loaded preprocessed data:")
        print(df.head())

    def load_glove_model(self):
        if os.path.exists(self.glove_path):
            print("Loading GloVe model from cache...")
            self.model = KeyedVectors.load(self.glove_path)
        else:
            print("Downloading GloVe model...")
            self.model = api.load("glove-twitter-200")
            self.model.save(self.glove_path)

    def process_tweet(self, cache_path=GLOVE_TWEET_VECTORS_CACHE_PATH):
        if self.original_data is None:
            self.load_orginal_data_and_preprocess_tweets()
        self.original_data['Timestamp'] = pd.to_datetime(
            self.original_data['Timestamp'], errors='coerce')

        # 检查转换后的数据是否有问题
        if self.original_data['Timestamp'].isnull().any():
            print(
                "Warning: Some timestamps could not be converted to datetime. These rows will be dropped.")
            self.original_data = self.original_data.dropna(
                subset=['Timestamp'])

        tweet_counts = self.original_data.groupby(
            ['MatchID', 'PeriodID']).size().reset_index(name='TweetCount')
        self.original_data = pd.merge(self.original_data, tweet_counts, on=[
            'MatchID', 'PeriodID'], how='right')

        # 计算每个比赛的开始时间
        match_start_times = self.original_data.groupby(
            'MatchID')['Timestamp'].min().reset_index()
        match_start_times.rename(
            columns={'Timestamp': 'MatchStartTime'}, inplace=True)

        # 计算每个阶段的开始时间
        period_start_times = self.original_data.groupby(['MatchID', 'PeriodID'])[
            'Timestamp'].min().reset_index()
        period_start_times.rename(
            columns={'Timestamp': 'PeriodStartTime'}, inplace=True)

        # 将比赛开始时间合并到阶段开始时间
        period_start_times = pd.merge(
            period_start_times, match_start_times, on='MatchID', how='left')

        # 计算持续时间
        period_start_times['DurationFromMatchStart'] = (
            period_start_times['PeriodStartTime'] - period_start_times['MatchStartTime']).dt.total_seconds()

        # 合并到原始数据
        period_durations = period_start_times[[
            'MatchID', 'PeriodID', 'DurationFromMatchStart']]
        self.original_data = pd.merge(self.original_data, period_durations, on=[
            'MatchID', 'PeriodID'], how='left')

        # 归一化 TweetCount 和 DurationFromMatchStart
        def normalize_group(group):
            scaler = MinMaxScaler()
            group[['TweetCount', 'DurationFromMatchStart']] = scaler.fit_transform(
                group[['TweetCount', 'DurationFromMatchStart']])
            return group

        self.original_data = self.original_data.groupby(
            'MatchID').apply(normalize_group)

        print("归一化后的数据：", self.original_data.head())

        if os.path.exists(cache_path):
            print("Loading tweet vectors from cache...")
            tweet_vectors = np.load(cache_path)
        else:
            print("Calculating tweet vectors...")
            tweet_vectors = np.vstack([self.word_to_vector(
                tweet) for tweet in self.original_data['Tweet']])
            np.save(cache_path,
                    tweet_vectors)  # 保存到 .npy 文件
            print(
                f"GloVe embeddings saved to {cache_path}")
        tweet_df = pd.DataFrame(tweet_vectors, index=self.original_data.index)
        self.original_data = pd.concat(
            [self.original_data, tweet_df], axis=1)
        self.original_data.to_csv(
            'preprocessed_original_data.csv', index=False)
        return np.array(tweet_vectors)

    def word_to_vector(self, tweet, vector_size=VECTOR_SIZE):
        words = tweet.split()
        word_vectors = [self.model[word]
                        for word in words if word in self.model]
        if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
            return np.zeros(vector_size)
        return np.mean(word_vectors, axis=0)

    def preprocess_text(self, text):  # Text preprocessing
        # normalize the text
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(' ', ' '))

        # Remove '#' from hashtags but keep the text
        text = re.sub(r'#', '', text)

        # Remove special characters except numbers, letters, plus and minus signs
        text = re.sub(r'[^\w\s\+\-]', '', text)
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)


print("------------------------------------------------------------------------------------")
# Convert tweets to GloVe embeddings
# 缓存文件路径

# period_tweets = self.original_data.groupby(['MatchID', 'PeriodID'])[
#     'Cleaned_Tweet'].apply(' '.join).reset_index()
# # Convert 'Timestamp' to datetime if it's not already
# self.original_data['Datetime'] = pd.to_datetime(
#     self.original_data['Timestamp'], unit='ms')

# # Extract the period start time by rounding down to the nearest minute
# self.original_data['Period_Start'] = self.original_data['Datetime'].dt.floor(
#     'T')

# # Calculate 'Time_Since_Period_Start' in seconds
# self.original_data['Time_Since_Period_Start'] = (
#     self.original_data['Datetime'] - self.original_data['Period_Start']).dt.total_seconds()
# # Group by 'MatchID' and 'PeriodID' to calculate temporal features
# temporal_features = self.original_data.groupby(['MatchID', 'PeriodID']).agg({
#     'Time_Since_Period_Start': ['mean', 'std'],
#     'ID': 'count'
# }).reset_index()

# # Flatten MultiIndex columns
# temporal_features.columns = [
#     'MatchID', 'PeriodID', 'Avg_Time_Since_Start', 'Std_Time_Since_Start', 'Tweet_Count']

# # Calculate 'Tweet_Rate' (tweets per second)
# temporal_features['Tweet_Rate'] = temporal_features['Tweet_Count'] / 60
# data = pd.merge(period_tweets, temporal_features,
#                 on=['MatchID', 'PeriodID'])


# if os.path.exists(PERIOD_FEATURES_CACHE_PATH):
#     print("Loading period features from cache...")
#     period_features = pd.read_parquet(PERIOD_FEATURES_CACHE_PATH)
# else:
#     print("Generating period features...")
#     tweet_df = pd.DataFrame(tweet_vectors)
#     period_features = pd.concat([df, tweet_df], axis=1)
#     period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
#     period_features = period_features.groupby(
#         ['MatchID', 'PeriodID', 'ID']).mean().reset_index()
#     period_features.to_parquet(PERIOD_FEATURES_CACHE_PATH, index=False)
#     print(f"Period features saved to {PERIOD_FEATURES_CACHE_PATH}")

# print("Period features are ready:")
# print(period_features.head())


myProcessor = Preprocessor()
myProcessor.process()


# # attribute value & target value
# # We drop the non-numerical features and keep the embeddings values for each period
# X = period_features.drop(
#     columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values

# # We extract the labels of our training samples
# y = period_features['EventType'].values


# print(X, y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42)

# # 初始化 XGBoost 模型
# model = XGBClassifier(n_estimators=400, max_depth=6,
#                       learning_rate=0.1, random_state=42)

# # 训练模型
# model.fit(X_train, y_train)

# # 预测和评估
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# print(classification_report(y_test, y_pred))


# predictions = []
# # We read each file separately, we preprocess the tweets and then use the classifier to predict the labels.
# # Finally, we concatenate all predictions into a list that will eventually be concatenated and exported
# # to be submitted on Kaggle.
# index = 0
# for fname in os.listdir("eval_tweets"):
#     val_df = pd.read_csv("eval_tweets/" + fname)
#     index += 1
#     val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

#     tweet_vectors = np.vstack([word_to_vector(
#         tweet, embeddings_model, VECTOR_SIZE) for tweet in val_df['Tweet']])
#     tweet_df = pd.DataFrame(tweet_vectors)

#     submit_features = pd.concat([val_df, tweet_df], axis=1)
#     submit_features = submit_features.drop(columns=['Timestamp', 'Tweet'])
#     submit_features = submit_features.groupby(
#         ['MatchID', 'PeriodID', 'ID']).mean().reset_index()
#     X = submit_features.drop(
#         columns=['MatchID', 'PeriodID', 'ID']).values
#     print(X[0]) if index == 1 else None
#     preds = model.predict(X)

#     submit_features['EventType'] = preds

#     predictions.append(submit_features[['ID', 'EventType']])

# pred_df = pd.concat(predictions)
# pred_df.to_csv('modal_predictions.csv', index=False)
