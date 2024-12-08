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

PROCESSED_DATA_CACHE_PATH = os.path.join(CACHE_DIR, "processed_data.csv")

# Load GloVe embeddings
print("start")


class Preprocessor:
    def __init__(self, glove_path=GLOVE_200_CACHE_PATH):
        self.glove_path = glove_path
        self.model = None
        self.original_data = None
        self.label = None
        self.features = None

    def process(self):
        if os.path.exists(PROCESSED_DATA_CACHE_PATH):
            print("Loading processed data from cache...")
            df = pd.read_csv(PROCESSED_DATA_CACHE_PATH)
            label = df['EventType'].values
            features = df.drop(columns=['EventType']).values
            return label, features
        self.load_glove_model()
        # if self.original_data is None:
        self.load_orginal_data_and_preprocess_tweets()
        self.process_tweet()
        return self.label, self.features

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
            # remove duplicate items
            df = df.drop_duplicates(
                subset=['MatchID', 'PeriodID', 'Tweet'], keep='first')
            # save it to cache file
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

        # 计算每个period的tweet数量
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
        self.original_data = self.original_data.drop(
            columns=['ID', 'Timestamp', 'Tweet'])
        y = self.original_data['EventType'].values
        X = self.original_data.drop(columns=['EventType']).values
        self.label = y
        self.features = X
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

eval_processor = Preprocessor()
(label, features) = eval_processor.process()
print("------------------------------------------------------------------------------------")
print(features)
print("------------------------------------------------------------------------------------")
print(label)


columns = ['MatchID', 'PeriodID', 'TweetCount',
           'DurationFromMatchStart'] + [str(i) for i in range(200)]
features_df = pd.DataFrame(features, columns=columns)
features_df['EventType'] = label
# 分组并计算推文向量均值
tweet_vector_columns = [str(i) for i in range(200)]
grouped = features_df.groupby(['MatchID', 'PeriodID'])

# 聚合推文向量（按均值）
aggregated_features = grouped[tweet_vector_columns].mean().reset_index()

# 合并数值特征
numerical_features = grouped[['TweetCount',
                              'DurationFromMatchStart']].first().reset_index()
aggregated_features = pd.merge(
    aggregated_features, numerical_features, on=['MatchID', 'PeriodID'])

# 合并目标变量
aggregated_features = pd.merge(
    aggregated_features,
    features_df[['MatchID', 'PeriodID', 'EventType']].drop_duplicates(),
    on=['MatchID', 'PeriodID']
)

# 准备训练数据
X = aggregated_features[tweet_vector_columns].values
y = aggregated_features['EventType'].values

print(X, y)

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# # 初始化 XGBoost 模型
model = XGBClassifier(n_estimators=1000, max_depth=5,
                      learning_rate=0.1, random_state=42)

# # 训练模型
model.fit(X_train, y_train)

# # 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


class PreprocessorForEvaluation:
    def __init__(self, glove_path=GLOVE_200_CACHE_PATH):
        self.glove_path = glove_path
        self.model = None
        self.original_data = None
        # self.label = None
        self.features = None

    def process(self):
        self.load_glove_model()
        self.load_orginal_data_and_preprocess_tweets()
        self.process_tweet()
        return self.features

    def load_orginal_data_and_preprocess_tweets(self):
        li = []
        print("reading file")
        for filename in os.listdir("eval_tweets"):
            df = pd.read_csv("eval_tweets/" + filename)
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
        # remove duplicate items
        df = df.drop_duplicates(
            subset=['MatchID', 'PeriodID', 'Tweet'], keep='first')
        self.original_data = df
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

        # 计算每个period的tweet数量
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

        print("Calculating tweet vectors...")
        tweet_vectors = np.vstack([self.word_to_vector(
            tweet) for tweet in self.original_data['Tweet']])
        tweet_df = pd.DataFrame(tweet_vectors, index=self.original_data.index)
        self.original_data = pd.concat(
            [self.original_data, tweet_df], axis=1)
        self.original_data = self.original_data.drop(
            columns=['Timestamp', 'Tweet'])
        X = self.original_data.values
        self.features = X
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


# 运行评估数据的处理器
eval_processor = PreprocessorForEvaluation()
features_withID = eval_processor.process()

# 定义列名
columns = ['ID', 'MatchID', 'PeriodID', 'TweetCount',
           'DurationFromMatchStart'] + [str(i) for i in range(200)]

# 将特征数组转化为 DataFrame
eval_features_df = pd.DataFrame(features_withID, columns=columns)
print("===================")
print(eval_features_df.head())
print("===================")

# 对数据按照 MatchID, PeriodID 分组
grouped = eval_features_df.groupby(['ID'])

# 计算推文向量特征的均值
tweet_vector_columns = [str(i) for i in range(200)]
aggregated_features = grouped[tweet_vector_columns].mean().reset_index()

# 获取数值特征 (TweetCount, DurationFromMatchStart)，这里使用 first() 是因为在同一个PeriodID下的数值是一致的
numerical_features = grouped[['TweetCount',
                              'DurationFromMatchStart']].first().reset_index()

# 合并均值后的特征与数值特征
aggregated_features = pd.merge(
    aggregated_features, numerical_features, on=['ID'])

print("aggregated_features", aggregated_features.head())

# 准备预测所需的数据 X_eval
X_eval = aggregated_features[[
    'TweetCount'] + tweet_vector_columns].values

# 使用训练好的模型进行预测
preds = model.predict(X_eval)

# 创建提交文件的 DataFrame
# 假设评价集中需要用 'MatchID' 作为输出的 ID 列，如果实际需要 'ID' 列，请确认哪个列是您需要输出的标识符。
submit_df = pd.DataFrame({
    'ID': aggregated_features['ID'],   # 如果您有单独的ID列，请换成对应的列
    'EventType': preds
})

# 导出结果到CSV
submit_df.to_csv('modal_predictions.csv', index=False)
